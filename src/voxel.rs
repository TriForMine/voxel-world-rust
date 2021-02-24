use building_blocks::storage::{IsEmpty, Sd16, SignedDistance};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct VoxelId(pub u8);

impl VoxelMaterial {
    pub const NULL: Self = Self(std::u8::MAX);
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Voxel {
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
