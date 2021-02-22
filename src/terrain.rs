use bevy::prelude::*;
use simdnoise::NoiseBuilder;

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app
            .add_startup_system(generate_terrain.system());
    }
}

fn generate_terrain(
    commands: &mut Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let size = 100;
    let noise = NoiseBuilder::ridge_2d(size, size)
        .with_freq(0.1)
        .with_seed(1234)
        .with_octaves(5)
        .generate_scaled(-1.0, 0.0);

    let mesh = meshes.add(Mesh::from(shape::Cube { size: 1.0 }));
    let material = materials.add(Color::GREEN.into());
    let yscale = 5.0f32;
    for x in 0..size {
        for z in 0..size {
            commands
                .spawn(PbrBundle {
                    transform: Transform::from_xyz(
                        x as f32,
                        (noise[x * size + z] * yscale).round() as f32,
                        z as f32,
                    ),
                    mesh: mesh.clone(),
                    material: material.clone(),
                    ..Default::default()
                });
        }
    }
}
