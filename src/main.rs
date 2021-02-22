use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::system::exit_on_esc_system,
    prelude::*,
};
use bevy_fly_camera::{
    FlyCamera,
    FlyCameraPlugin,
};

use voxel_world_rust::{
    terrain::*,
};

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FlyCameraPlugin)
        .add_plugin(TerrainPlugin)
        .add_system(exit_on_esc_system.system())
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    commands: &mut Commands,
) {
    commands
        .spawn(LightBundle {
            transform: Transform::from_xyz(14.0, 18.0, 14.0),
            ..Default::default()
        })
        .spawn(PerspectiveCameraBundle {
            transform: Transform::from_xyz(10f32, 10f32, -10f32),
            ..Default::default()
        })
        .with(FlyCamera::default());
}
