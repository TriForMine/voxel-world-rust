use crate::camera::CameraTag;
use crate::voxel::{Voxel, VoxelId, VoxelIdInfo, VoxelMaterial};
use bevy::{prelude::*, render::texture::AddressMode};
use bevy_building_blocks::{
    empty_compressible_chunk_map, ChunkCacheConfig, MapIoPlugin, VoxelEditor, VoxelMap,
    VoxelPalette,
};
use building_blocks::{core::prelude::*, storage::Sd16};
#[cfg(feature = "simd")]
use simdnoise::*;
use smooth_voxel_renderer::{
    ArrayMaterial, MeshGeneratorPlugin, MeshMaterial, SmoothVoxelRenderPlugin,
};

const CHUNK_SIZE: usize = 32;
const SEA_LEVEL: i32 = 42;
const VIEW_DISTANCE: usize = 8;
const Y_SCALE: f32 = 64.0 * 8.0;

struct MeshesResource {
    pub generated_map: Vec<Point3i>,
}

impl Default for MeshesResource {
    fn default() -> Self {
        MeshesResource {
            generated_map: Vec::new(),
        }
    }
}

fn get_block_by_height(y: i32, max_y: i32) -> VoxelId {
    if y == max_y {
        VoxelId(2)
    } else if y > max_y - 4 {
        VoxelId(1)
    } else {
        VoxelId(3)
    }
}

fn generate_chunk(x: i32, z: i32, voxel_editor: &mut VoxelEditor<Voxel>) {
    let (noise, _min, max) =
        simdnoise::NoiseBuilder::gradient_2d_offset(x as f32, CHUNK_SIZE, z as f32, CHUNK_SIZE)
            .with_freq(0.008)
            .with_seed(15464)
            .generate();


    let min = PointN([x, 0, z]);
    let max = PointN([
        x + CHUNK_SIZE as i32 - 1,
        SEA_LEVEL + (Y_SCALE * max).round() as i32,
        z + CHUNK_SIZE as i32 - 1,
    ]);

    let (cave_noise, _min, _cave_max) =
        simdnoise::NoiseBuilder::turbulence_3d_offset(x as f32, CHUNK_SIZE, z as f32, CHUNK_SIZE, 0.0, max.y() as usize)
            .with_freq(0.02)
            .with_seed(15464)
            .generate();

    voxel_editor.edit_extent_and_touch_neighbors(
        Extent3i::from_min_and_max(min, max),
        |p, voxel| {
            let height = SEA_LEVEL
                + (noise[(p.z() - z) as usize * CHUNK_SIZE + (p.x() - x) as usize] * Y_SCALE)
                .round() as i32;
            *voxel = if p.y() <= height {
                let cave_value = cave_noise[(p.x() - x) as usize + CHUNK_SIZE * p.y() as usize + CHUNK_SIZE * CHUNK_SIZE * (p.z() - z) as usize] * 64.0;
                if cave_value < 1.0 {
                    Voxel::new(VoxelId(0), Sd16::ONE)
                } else {
                    Voxel::new(get_block_by_height(p.y(), height as i32), Sd16::NEG_ONE)
                }
            } else {
                Voxel::new(VoxelId(0), Sd16::ONE)
            }
        },
    );
}

fn generate_voxels(
    mut voxel_editor: VoxelEditor<Voxel>,
    mut res: ResMut<MeshesResource>,
    query: Query<&Transform, With<CameraTag>>,
) {
    for cam_transform in query.iter() {
        let cam_pos = cam_transform.translation;
        let cam_pos = PointN([cam_pos.x.round() as i32, 0i32, cam_pos.z.round() as i32]);

        let extent = transform_to_extent(cam_pos, (VIEW_DISTANCE * CHUNK_SIZE) as i32);
        let extent = extent_modulo_expand(extent, CHUNK_SIZE as i32);
        let min = extent.minimum;
        let max = extent.least_upper_bound();

        for x in (min.x()..max.x()).step_by(CHUNK_SIZE as usize) {
            for z in (min.z()..max.z()).step_by(CHUNK_SIZE as usize) {
                let chunk_pos = PointN([x as i32, 0, z as i32]);
                if res.generated_map.contains(&chunk_pos) {
                    continue;
                }
                generate_chunk(x as i32, z as i32, &mut voxel_editor);
                res.generated_map.push(chunk_pos);
            }
        }
    }
}

fn transform_to_extent(cam_pos: Point3i, view_distance: i32) -> Extent3i {
    Extent3i::from_min_and_lub(
        PointN([cam_pos.x() - view_distance, 0, cam_pos.z() - view_distance]),
        PointN([cam_pos.x() + view_distance, 0, cam_pos.z() + view_distance]),
    )
}

fn modulo_down(v: i32, modulo: i32) -> i32 {
    (v / modulo) * modulo
}

fn modulo_up(v: i32, modulo: i32) -> i32 {
    ((v / modulo) + 1) * modulo
}

fn extent_modulo_expand(extent: Extent3i, modulo: i32) -> Extent3i {
    let min = extent.minimum;
    let max = extent.least_upper_bound();
    Extent3i::from_min_and_lub(
        PointN([
            modulo_down(min.x(), modulo),
            min.y(),
            modulo_down(min.z(), modulo),
        ]),
        PointN([
            modulo_up(max.x(), modulo) + 1,
            max.y() + 1,
            modulo_up(max.z(), modulo) + 1,
        ]),
    )
}

struct LoadingTexture(Handle<Texture>);

fn init_materials(commands: &mut Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(100.0, 100.0, 100.0)),
            ..Default::default()
        })
        .insert_resource(VoxelMap {
            voxels: empty_compressible_chunk_map::<Voxel>(PointN([CHUNK_SIZE as i32; 3])),
            palette: VoxelPalette {
                infos: vec![
                    VoxelIdInfo {
                        is_empty: true,
                        material: VoxelMaterial::NULL,
                    },
                    VoxelIdInfo {
                        is_empty: false,
                        material: VoxelMaterial(0),
                    },
                    VoxelIdInfo {
                        is_empty: false,
                        material: VoxelMaterial(1),
                    },
                    VoxelIdInfo {
                        is_empty: false,
                        material: VoxelMaterial(2),
                    },
                    VoxelIdInfo {
                        is_empty: false,
                        material: VoxelMaterial(3),
                    },
                ],
            },
        })
        .insert_resource(LoadingTexture(asset_server.load("spritesheet.png")));
}

fn prepare_materials_texture(texture: &mut Texture) {
    let num_layers = 4;
    texture.reinterpret_stacked_2d_as_array(num_layers);
    texture.sampler.address_mode_u = AddressMode::Repeat;
    texture.sampler.address_mode_v = AddressMode::Repeat;
}

fn wait_for_assets_loaded(
    commands: &mut Commands,
    loading_texture: Res<LoadingTexture>,
    mut array_materials: ResMut<Assets<ArrayMaterial>>,
    mut textures: ResMut<Assets<Texture>>,
    mut state: ResMut<State<GameState>>,
) {
    if let Some(texture) = textures.get_mut(&loading_texture.0) {
        println!("Done loading mesh texture");
        prepare_materials_texture(texture);
        commands.insert_resource(MeshMaterial(
            array_materials.add(ArrayMaterial::from(loading_texture.0.clone())),
        ));
        state.set_next(GameState::Playing).unwrap();
    }
}

pub struct TerrainPlugin;

#[derive(Clone)]
enum GameState {
    Loading,
    Playing,
}

mod stages {
    pub const GAME_STAGE: &str = "game";
}

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource::<MeshesResource>(MeshesResource::default())
            .add_startup_system(MeshGeneratorPlugin::<Voxel>::initialize.system())
            .add_plugin(SmoothVoxelRenderPlugin)
            // This plugin should run systems in the LAST stage.
            .add_plugin(MapIoPlugin::<Voxel>::new(
                PointN([CHUNK_SIZE as i32; 3]),
                ChunkCacheConfig::default(),
            ));

        add_game_schedule(app);
    }
}

fn add_game_schedule(app: &mut AppBuilder) {
    let mut game_state_stage = StateStage::<GameState>::default();
    game_state_stage
        .on_state_enter(GameState::Loading, init_materials.system())
        .on_state_update(GameState::Loading, wait_for_assets_loaded.system())
        .enter_stage(GameState::Playing, |stage: &mut SystemStage| {
            stage
                .add_system(generate_voxels.system())
                .add_system(MeshGeneratorPlugin::<Voxel>::initialize.system())
        })
        .update_stage(GameState::Playing, |stage: &mut SystemStage| {
            MeshGeneratorPlugin::<Voxel>::update_in_stage(stage);

            stage.add_system(generate_voxels.system())
        });

    app.insert_resource(State::new(GameState::Loading))
        .add_stage_after(stage::UPDATE, stages::GAME_STAGE, game_state_stage);
}
