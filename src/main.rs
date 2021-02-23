use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::{keyboard::KeyCode, system::exit_on_esc_system},
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
        .add_startup_system(setup.system())
        .run();
}

fn setup(commands: &mut Commands) {
    commands.spawn(PbrBundle {
        transform: Transform::from_translation(Vec3::new(14.0, 18.0, 14.0)),
        ..Default::default()
    });
}
