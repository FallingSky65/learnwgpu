use std::{
    collections::HashMap,
    ops::DerefMut,
    sync::{Arc, Mutex, Weak},
};

use wgpu::util::DeviceExt;

use crate::{
    model::{Material, Mesh, Model, ModelVertex},
    resources,
};

#[derive(Clone, Copy)]
enum BlockType {
    AIR,
    GRASS,
    DIRT,
}

#[derive(Clone, Copy)]
struct Block {
    block_type: BlockType,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            block_type: BlockType::AIR,
        }
    }
}

struct Terrain {}

impl Terrain {
    fn evaluate(x: i32, y: i32, z: i32) -> BlockType {
        if y < 1 {
            BlockType::DIRT
        } else if y == 1 {
            BlockType::GRASS
        } else {
            BlockType::AIR
        }
    }
}

const CHUNK_W: usize = 16;
const CHUNK_H: usize = 64;
const CHUNK_L: usize = 16;
pub struct Chunk {
    world_x: i32,
    world_y: i32,
    world_z: i32,
    chunk_data: [[[Block; CHUNK_L]; CHUNK_H]; CHUNK_W],
    chunk_mesh: Option<Arc<Mesh>>,
    world: Weak<World>,
}

impl Chunk {
    fn new(x: i32, y: i32, z: i32, world: &Arc<World>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            world_x: x,
            world_y: y,
            world_z: z,
            chunk_data: [[[Block::default(); CHUNK_L]; CHUNK_H]; CHUNK_W],
            chunk_mesh: None,
            world: Arc::downgrade(world),
        }))
    }

    pub fn gen_mesh(&mut self, device: &wgpu::Device) {
        let vertices = [
            // Top Face
            ModelVertex {
                position: [0.0, 1.0, 0.0],
                tex_coords: [0.0, 1.0],
                normal: [0.0, 1.0, 0.0],
            },
            ModelVertex {
                position: [0.0, 1.0, 1.0],
                tex_coords: [1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
            },
            ModelVertex {
                position: [1.0, 1.0, 1.0],
                tex_coords: [1.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
            ModelVertex {
                position: [1.0, 1.0, 0.0],
                tex_coords: [0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            },
            // +Z Face
            ModelVertex {
                position: [0.0, 0.0, 1.0],
                tex_coords: [1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
            },
            ModelVertex {
                position: [1.0, 0.0, 1.0],
                tex_coords: [2.0, 1.0],
                normal: [0.0, 0.0, 1.0],
            },
            ModelVertex {
                position: [1.0, 1.0, 1.0],
                tex_coords: [2.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            ModelVertex {
                position: [0.0, 1.0, 1.0],
                tex_coords: [1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            // +X Face
            ModelVertex {
                position: [1.0, 0.0, 1.0],
                tex_coords: [1.0, 1.0],
                normal: [1.0, 0.0, 0.0],
            },
            ModelVertex {
                position: [1.0, 0.0, 0.0],
                tex_coords: [2.0, 1.0],
                normal: [1.0, 0.0, 0.0],
            },
            ModelVertex {
                position: [1.0, 1.0, 0.0],
                tex_coords: [2.0, 0.0],
                normal: [1.0, 0.0, 0.0],
            },
            ModelVertex {
                position: [1.0, 1.0, 1.0],
                tex_coords: [1.0, 0.0],
                normal: [1.0, 0.0, 0.0],
            },
            // -Z Face
            ModelVertex {
                position: [1.0, 0.0, 0.0],
                tex_coords: [1.0, 1.0],
                normal: [0.0, 0.0, -1.0],
            },
            ModelVertex {
                position: [0.0, 0.0, 0.0],
                tex_coords: [2.0, 1.0],
                normal: [0.0, 0.0, -1.0],
            },
            ModelVertex {
                position: [0.0, 1.0, 0.0],
                tex_coords: [2.0, 0.0],
                normal: [0.0, 0.0, -1.0],
            },
            ModelVertex {
                position: [1.0, 1.0, 0.0],
                tex_coords: [1.0, 0.0],
                normal: [0.0, 0.0, -1.0],
            },
            // -X Face
            ModelVertex {
                position: [0.0, 0.0, 0.0],
                tex_coords: [1.0, 1.0],
                normal: [-1.0, 0.0, 0.0],
            },
            ModelVertex {
                position: [0.0, 0.0, 1.0],
                tex_coords: [2.0, 1.0],
                normal: [-1.0, 0.0, 0.0],
            },
            ModelVertex {
                position: [0.0, 1.0, 1.0],
                tex_coords: [2.0, 0.0],
                normal: [-1.0, 0.0, 0.0],
            },
            ModelVertex {
                position: [0.0, 1.0, 0.0],
                tex_coords: [1.0, 0.0],
                normal: [-1.0, 0.0, 0.0],
            },
            // Bottom Face
            ModelVertex {
                position: [0.0, 0.0, 0.0],
                tex_coords: [2.0, 1.0],
                normal: [0.0, -1.0, 0.0],
            },
            ModelVertex {
                position: [1.0, 0.0, 0.0],
                tex_coords: [3.0, 1.0],
                normal: [0.0, -1.0, 0.0],
            },
            ModelVertex {
                position: [1.0, 0.0, 1.0],
                tex_coords: [3.0, 0.0],
                normal: [0.0, -1.0, 0.0],
            },
            ModelVertex {
                position: [0.0, 0.0, 1.0],
                tex_coords: [2.0, 0.0],
                normal: [0.0, -1.0, 0.0],
            },
        ];

        #[rustfmt::skip]
        let indices: [u32; 36] = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!(
                "Mesh Vertex Buffer at ({}, {}, {})",
                self.world_x, self.world_y, self.world_z
            )),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!(
                "Mesh Index Buffer at ({}, {}, {})",
                self.world_x, self.world_y, self.world_z
            )),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.chunk_mesh = Some(Arc::new(Mesh {
            name: format!(
                "Chunk Mesh at ({}, {}, {})",
                self.world_x, self.world_y, self.world_z
            ),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0,
        }))
    }
}

pub struct World {
    pub chunks: Mutex<HashMap<(i32, i32), Arc<Mutex<Chunk>>>>,
    material: Arc<Material>,
    terrain: Terrain,
}

impl World {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> anyhow::Result<Arc<Self>> {
        let chunks = Mutex::new(HashMap::new());

        let diffuse_texture = resources::load_texture("Grass.png", device, queue).await?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: None,
        });

        let material = Arc::new(Material {
            name: "mc material".to_string(),
            diffuse_texture,
            bind_group,
        });

        let terrain = Terrain {};

        Ok(Arc::new(Self {
            chunks,
            material,
            terrain,
        }))
    }

    pub fn gen_chunk(self: &Arc<Self>, x: i32, y: i32, z: i32) -> Arc<Mutex<Chunk>> {
        let chunk = Chunk::new(x, y, z, self);
        self.chunks.lock().unwrap().insert((x, z), chunk.clone());
        chunk
    }

    pub fn get_model(&self) -> Model {
        let mut meshes: Vec<Weak<Mesh>> = Vec::new();
        let materials: Vec<Weak<Material>> = vec![Arc::downgrade(&self.material)];

        meshes.push(Arc::downgrade(
            self.chunks
                .lock()
                .unwrap()
                .get(&(0, 0))
                .unwrap()
                .lock()
                .unwrap()
                .chunk_mesh
                .as_ref()
                .unwrap(),
        ));

        Model { meshes, materials }
    }
}
