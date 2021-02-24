use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::system::exit_on_esc_system,
    prelude::*,
};
use bevy_flycam::PlayerPlugin;

use voxel_world_rust::terrain::*;

fn main() {
    App::build()
        .insert_resource(WindowDescriptor {
            title: "Voxel World Rust".to_string(),
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(PlayerPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(TerrainPlugin)
        .add_system(exit_on_esc_system.system())
        .run();
}
