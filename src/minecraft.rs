use std::{
    collections::HashMap,
    ops::DerefMut,
    sync::{Arc, Mutex, Weak},
};

use num_enum::IntoPrimitive;
use wgpu::util::DeviceExt;

use crate::{
    model::{Material, Mesh, Model, Vertex},
    resources,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MCVertex {
    pub position: [f32; 3],
    pub id: u32,
    // id & 0x00FF: vertex id
    // id & 0xFF00: block id
}

impl MCVertex {
    fn new(position: [u32; 3], offset: [u32; 3], id: u32) -> Self {
        MCVertex {
            position: [
                (position[0] + offset[0]) as f32,
                (position[1] + offset[1]) as f32,
                (position[2] + offset[2]) as f32,
            ],
            id,
        }
    }
}

impl Vertex for MCVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<MCVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, IntoPrimitive, PartialEq, Eq)]
enum BlockId {
    AIR,
    GRASS,
    DIRT,
}

#[derive(Clone, Copy)]
struct Block {
    block_type: BlockId,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            block_type: BlockId::AIR,
        }
    }
}

struct Terrain {}

impl Terrain {
    fn evaluate(x: i32, y: i32, z: i32) -> BlockId {
        if y < 1 {
            BlockId::DIRT
        } else if y == 1 {
            BlockId::GRASS
        } else {
            BlockId::AIR
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

fn add_face(
    vertices: &mut Vec<MCVertex>,
    indices: &mut Vec<u32>,
    offset: [u32; 3],
    block_id: BlockId,
    face_id: u32,
) {
    if block_id == BlockId::AIR {
        return;
    }
    let block_id = (block_id as u32) << 8;
    let i_start = vertices.len() as u32;
    match face_id {
        0 => {
            vertices.push(MCVertex::new([0, 0, 0], offset, block_id | 0));
            vertices.push(MCVertex::new([0, 0, 1], offset, block_id | 1));
            vertices.push(MCVertex::new([0, 1, 1], offset, block_id | 2));
            vertices.push(MCVertex::new([0, 1, 0], offset, block_id | 3));
        }
        1 => {
            vertices.push(MCVertex::new([0, 0, 0], offset, block_id | 4));
            vertices.push(MCVertex::new([1, 0, 0], offset, block_id | 5));
            vertices.push(MCVertex::new([1, 0, 1], offset, block_id | 6));
            vertices.push(MCVertex::new([0, 0, 1], offset, block_id | 7));
        }
        2 => {
            vertices.push(MCVertex::new([0, 0, 0], offset, block_id | 8));
            vertices.push(MCVertex::new([0, 1, 0], offset, block_id | 9));
            vertices.push(MCVertex::new([1, 1, 0], offset, block_id | 10));
            vertices.push(MCVertex::new([1, 0, 0], offset, block_id | 11));
        }
        3 => {
            vertices.push(MCVertex::new([1, 0, 0], offset, block_id | 12));
            vertices.push(MCVertex::new([1, 1, 0], offset, block_id | 13));
            vertices.push(MCVertex::new([1, 1, 1], offset, block_id | 14));
            vertices.push(MCVertex::new([1, 0, 1], offset, block_id | 15));
        }
        4 => {
            vertices.push(MCVertex::new([0, 1, 0], offset, block_id | 16));
            vertices.push(MCVertex::new([0, 1, 1], offset, block_id | 17));
            vertices.push(MCVertex::new([1, 1, 1], offset, block_id | 18));
            vertices.push(MCVertex::new([1, 1, 0], offset, block_id | 19));
        }
        5 => {
            vertices.push(MCVertex::new([0, 0, 1], offset, block_id | 20));
            vertices.push(MCVertex::new([1, 0, 1], offset, block_id | 21));
            vertices.push(MCVertex::new([1, 1, 1], offset, block_id | 22));
            vertices.push(MCVertex::new([0, 1, 1], offset, block_id | 23));
        }
        _ => return,
    }
    indices.push(i_start);
    indices.push(i_start + 1);
    indices.push(i_start + 2);
    indices.push(i_start + 2);
    indices.push(i_start + 3);
    indices.push(i_start);
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
        let mut vertices: Vec<MCVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        println!("num vertices {}", vertices.len());
        println!("num indices {}", indices.len());
        add_face(&mut vertices, &mut indices, [0, 0, 0], BlockId::GRASS, 0);
        add_face(&mut vertices, &mut indices, [0, 0, 0], BlockId::GRASS, 1);
        add_face(&mut vertices, &mut indices, [0, 0, 0], BlockId::GRASS, 2);
        add_face(&mut vertices, &mut indices, [0, 0, 0], BlockId::GRASS, 3);
        add_face(&mut vertices, &mut indices, [0, 0, 0], BlockId::GRASS, 4);
        add_face(&mut vertices, &mut indices, [0, 0, 0], BlockId::GRASS, 5);
        println!("num vertices {}", vertices.len());
        println!("num indices {}", indices.len());

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
