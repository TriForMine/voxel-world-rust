use bevy::{
    prelude::*,
    render::{prelude::*, texture::AddressMode},
};
use bevy_building_blocks::{
    empty_compressible_chunk_map, ChunkCacheConfig, MapIoPlugin, VoxelEditor, VoxelMap,
    VoxelPalette,
};
use building_blocks::{
    core::prelude::*,
    storage::{IsEmpty, Sd16, SignedDistance},
};
use core::arch::x86_64::{_mm256_set1_ps, _mm256_storeu_ps};
use serde::{Deserialize, Serialize};
#[cfg(feature = "simd")]
use simdnoise::*;
use smooth_voxel_renderer::{
    ArrayMaterial, LightBundle, MeshGeneratorPlugin, MeshMaterial, SmoothVoxelRenderPlugin,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VoxelId(pub u8);

impl VoxelMaterial {
    pub const NULL: Self = Self(std::u8::MAX);
}

const CHUNK_SIZE: usize = 32;
const SEA_LEVEL: i32 = 42;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Voxel {
    pub voxel_id: VoxelId,
    pub distance: Sd16,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VoxelIdInfo {
    pub is_empty: bool,
    pub material: VoxelMaterial,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VoxelMaterial(pub u8);

impl Voxel {
    pub fn new(voxel_id: VoxelId, distance: Sd16) -> Self {
        Self { voxel_id, distance }
    }
}

impl Default for Voxel {
    fn default() -> Self {
        Voxel {
            voxel_id: VoxelId(0),
            distance: Sd16::ONE,
        }
    }
}

impl SignedDistance for Voxel {
    fn is_negative(self) -> bool {
        self.distance.0 < 0
    }
}

impl From<Voxel> for f32 {
    fn from(v: Voxel) -> f32 {
        v.distance.into()
    }
}

impl IsEmpty for &VoxelIdInfo {
    fn is_empty(&self) -> bool {
        self.is_empty
    }
}

impl smooth_voxel_renderer::MaterialVoxel for &VoxelIdInfo {
    fn material(&self) -> smooth_voxel_renderer::MaterialLayer {
        smooth_voxel_renderer::MaterialLayer(self.material.0)
    }
}

impl bevy_building_blocks::Voxel for Voxel {
    type TypeInfo = VoxelIdInfo;

    fn get_type_index(&self) -> usize {
        self.voxel_id.0 as usize
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

fn get_block_by_height(y: i32, max_y: i32) -> VoxelId {
    if y == max_y {
        VoxelId(2)
    } else if y > max_y - 4 {
        VoxelId(1)
    } else {
        VoxelId(3)
    }
}

fn generate_voxels(mut voxel_editor: VoxelEditor<Voxel>) {
    println!("Initializing voxels");

    let generated_chunks = 6;
    let yscale = 8.0f32;

    let noise = simdnoise::NoiseBuilder::gradient_2d(
        generated_chunks * CHUNK_SIZE,
        generated_chunks * CHUNK_SIZE,
    )
    .with_freq(0.008)
    .with_seed(15464)
    .generate_scaled(-1.0, 1.0);

    for x in (0..generated_chunks * CHUNK_SIZE).step_by(CHUNK_SIZE as usize) {
        for z in (0..generated_chunks * CHUNK_SIZE).step_by(CHUNK_SIZE as usize) {
            let min = PointN([x as i32, 0, z as i32]);
            let max = PointN([
                x as i32 + CHUNK_SIZE as i32 - 1,
                SEA_LEVEL + yscale as i32,
                z as i32 + CHUNK_SIZE as i32 - 1,
            ]);
            voxel_editor.edit_extent(Extent3i::from_min_and_max(min, max), |p, voxel| {
                let height = SEA_LEVEL
                    + (noise[p.z() as usize * generated_chunks * CHUNK_SIZE + p.x() as usize]
                        * yscale)
                        .round() as i32;
                *voxel = if p.y() <= height {
                    Voxel::new(get_block_by_height(p.y(), height as i32), Sd16::NEG_ONE)
                } else {
                    Voxel::new(VoxelId(0), Sd16::ONE)
                }
            });
        }
    }
    println!("Voxels initialized");
}

struct LoadingTexture(Handle<Texture>);

fn init_materials(commands: &mut Commands, asset_server: Res<AssetServer>) {
    commands
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

            stage
        });

    app.insert_resource(State::new(GameState::Loading))
        .add_stage_after(stage::UPDATE, stages::GAME_STAGE, game_state_stage);
}
