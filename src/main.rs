use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::{keyboard::KeyCode, system::exit_on_esc_system},
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
        .insert_resource(WindowDescriptor {
            title: "Voxel World Rust".to_string(),
            ..Default::default()
        })
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Msaa { samples: 4 })
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
            transform: Transform::from_translation(Vec3::new(14.0, 18.0, 14.0)),
            ..Default::default()
        })
        .spawn(PerspectiveCameraBundle {
            transform: Transform::from_xyz(10f32, 10f32, -10f32),
            ..Default::default()
        })
        .with(FlyCamera {
            pitch: 40.0f32,
            yaw: -135.0f32,
            key_up: KeyCode::A,
            key_down: KeyCode::E,
            key_left: KeyCode::Q,
            key_forward: KeyCode::Z,
            ..Default::default()
        });
}
