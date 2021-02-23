use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
    tasks::{ComputeTaskPool, TaskPool},
};
use building_blocks::{
    core::prelude::*,
    mesh::*,
    storage::{prelude::*, IsEmpty},
};
use core::arch::x86_64::{_mm256_set1_ps, _mm256_storeu_ps};
#[cfg(feature = "simd")]
use simdnoise::simd;
use std::collections::{HashMap, HashSet};

type VoxelMap = ChunkHashMap3<Voxel>;
type VoxelId = u8;

const SEA_LEVEL: i32 = 64;
const MAX_CHUNK: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Voxel(VoxelId);

impl Default for Voxel {
    fn default() -> Self {
        Voxel(0)
    }
}

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl IsOpaque for Voxel {
    fn is_opaque(&self) -> bool {
        true
    }
}

#[derive(Eq, PartialEq)]
struct TrivialMergeValue;

impl MergeVoxel for Voxel {
    type VoxelValue = TrivialMergeValue;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        TrivialMergeValue
    }
}

struct VoxelResource {
    pub map: VoxelMap,
    pub chunk_size: i32,
    pub materials: Vec<Handle<StandardMaterial>>,
}

impl Default for VoxelResource {
    fn default() -> Self {
        let chunk_size = 32;
        VoxelResource {
            chunk_size: chunk_size as i32,
            map: ChunkMapBuilder3 {
                chunk_shape: PointN([chunk_size; 3]),
                ambient_value: Voxel(0),
                default_chunk_metadata: (),
            }
            .build_with_hash_map_storage(),
            materials: Vec::new(),
        }
    }
}

struct MeshesResource {
    pub generated_map: HashMap<Point3i, Vec<Entity>>,
}

impl Default for MeshesResource {
    fn default() -> Self {
        MeshesResource {
            generated_map: HashMap::new(),
        }
    }
}

fn generate_chunk(res: &mut ResMut<VoxelResource>, min: Point3i, max: Point3i) {
    let yscale = 128.0f32;
    for x in min.z()..max.z() {
        for z in min.x()..max.x() {
            unsafe {
                let max_y = SEA_LEVEL
                    + (if is_x86_feature_detected!("avx2") {
                        let x_256 = _mm256_set1_ps((x as f32) * 0.02);
                        let z_256 = _mm256_set1_ps((z as f32) * 0.02);
                        let s = simdnoise::avx2::simplex_2d(x_256, z_256, 4654545);
                        let mut r: f32 = 0.0;
                        _mm256_storeu_ps(&mut r, s);
                        r
                    } else {
                        simdnoise::scalar::simplex_2d((x as f32) * 0.1, (z as f32) * 0.1, 4654545)
                    } * yscale)
                        .round() as i32;
                for y in 0..(max_y + 1) {
                    *res.map.get_mut(PointN([x, y, z])) = Voxel(1);
                }
            }
        }
    }
}

fn generate_voxels(mut voxels: ResMut<VoxelResource>, voxel_meshes: Res<MeshesResource>) {
    for z in (0..128).step_by(voxels.chunk_size as usize) {
        for x in (0..128).step_by(voxels.chunk_size as usize) {
            let p = PointN([x, 0, z]);
            if voxel_meshes.generated_map.get(&p).is_some() {
                return;
            }
            let chunk_size = voxels.chunk_size;
            generate_chunk(
                &mut voxels,
                PointN([x, 0, z]),
                PointN([x + chunk_size, 64, z + chunk_size]),
            );
        }
    }
}

fn get_mesh(voxel_map: &VoxelMap, pool: &TaskPool) -> Vec<Option<PosNormMesh>> {
    pool.scope(|s| {
        for chunk_key in voxel_map.storage().keys() {
            s.spawn(async move {
                let padded_chunk_extent = padded_greedy_quads_chunk_extent(
                    &voxel_map.indexer.extent_for_chunk_at_key(*chunk_key),
                );

                let mut padded_chunk = Array3::fill(padded_chunk_extent, Voxel(0));
                copy_extent(&padded_chunk_extent, voxel_map, &mut padded_chunk);

                let mut buffer = GreedyQuadsBuffer::new(padded_chunk_extent);
                greedy_quads(&padded_chunk, &padded_chunk_extent, &mut buffer);

                let mut mesh = PosNormMesh::default();
                for group in buffer.quad_groups.iter() {
                    for quad in group.quads.iter() {
                        group.face.add_quad_to_pos_norm_mesh(&quad, &mut mesh);
                    }
                }

                if mesh.is_empty() {
                    None
                } else {
                    Some(mesh)
                }
            })
        }
    })
}

fn create_mesh_entity(
    mesh: PosNormMesh,
    commands: &mut Commands,
    material: Handle<StandardMaterial>,
    meshes: &mut Assets<Mesh>,
) -> Entity {
    assert_eq!(mesh.positions.len(), mesh.normals.len());
    let num_vertices = mesh.positions.len();

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.set_attribute(
        "Vertex_Position",
        VertexAttributeValues::Float3(mesh.positions),
    );
    render_mesh.set_attribute("Vertex_Normal", VertexAttributeValues::Float3(mesh.normals));
    render_mesh.set_attribute(
        "Vertex_Uv",
        VertexAttributeValues::Float2(vec![[0.0; 2]; num_vertices]),
    );
    render_mesh.set_indices(Some(Indices::U32(mesh.indices)));

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(render_mesh),
            material,
            ..Default::default()
        })
        .current_entity()
        .unwrap()
}

fn generate_meshes(
    commands: &mut Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    voxels: ChangedRes<VoxelResource>,
    mut voxel_meshes: ResMut<MeshesResource>,
    pool: Res<ComputeTaskPool>,
) {
    let mut to_remove: HashSet<Point3i> = voxel_meshes.generated_map.keys().cloned().collect();
    for x in 0..8 {
        for z in 0..8 {
            let p = PointN([x, 0, z]);
            to_remove.remove(&p);
            if voxel_meshes.generated_map.get(&p).is_some() {
                return;
            }
            let entity_mesh = get_mesh(&voxels.map, &pool);
            let mut generated_meshes: Vec<Entity> = Vec::new();
            for mesh in entity_mesh.into_iter() {
                if let Some(mesh) = mesh {
                    generated_meshes.push(create_mesh_entity(
                        mesh,
                        commands,
                        voxels.materials[1].clone(),
                        &mut meshes,
                    ))
                }
            }
            voxel_meshes.generated_map.insert(p, generated_meshes);
        }
    }
    for p in &to_remove {
        if let Some(entities) = voxel_meshes.generated_map.remove(p) {
            for entity in entities {
                commands.despawn(entity);
            }
        }
    }
}

fn init_materials(
    mut res: ResMut<VoxelResource>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let dirt_texture = asset_server.load("blocks/dirt.png");

    res.materials.push(materials.add(Color::NONE.into()));
    res.materials.push(materials.add(StandardMaterial {
        albedo_texture: Some(dirt_texture),
        unlit: true,
        ..Default::default()
    }));
}

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource::<VoxelResource>(VoxelResource::default())
            .insert_resource::<MeshesResource>(MeshesResource::default())
            .add_startup_system(init_materials.system())
            .add_system(generate_voxels.system())
            .add_system(generate_meshes.system());
    }
}
