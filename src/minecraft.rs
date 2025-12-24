use std::{
    collections::{HashMap, HashSet},
    ops::DerefMut,
    sync::{Arc, Mutex, MutexGuard, PoisonError, Weak},
};

use fastnoise_lite::FastNoiseLite;
use num_enum::IntoPrimitive;
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
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
    fn new(position: [u32; 3], offset: [i32; 3], id: u32) -> Self {
        MCVertex {
            position: [
                (position[0] as i32 + offset[0]) as f32,
                (position[1] as i32 + offset[1]) as f32,
                (position[2] as i32 + offset[2]) as f32,
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

struct Terrain {
    n1: FastNoiseLite,
    n2: FastNoiseLite,
    n3: FastNoiseLite,
    n4: FastNoiseLite,
}

impl Terrain {
    fn new() -> Self {
        let mut n1 = FastNoiseLite::new();
        n1.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
        let mut n2 = FastNoiseLite::new();
        n2.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
        let mut n3 = FastNoiseLite::new();
        n3.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
        let mut n4 = FastNoiseLite::new();
        n4.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
        Terrain { n1, n2, n3, n4 }
    }

    fn evaluate(&self, x: i32, y: i32, z: i32) -> BlockId {
        let xf = x as f32 / 2.0;
        let yf = y as f32 / 2.0;
        let zf = z as f32 / 2.0;
        let h = 60.0
            * ((0.5 * self.n1.get_noise_2d(xf, zf) + 0.5) / 2.0
                + (0.5 * self.n2.get_noise_2d(xf * 2.0, zf * 2.0) + 0.5) / 4.0
                + (0.5 * self.n3.get_noise_2d(xf * 4.0, zf * 4.0) + 0.5) / 8.0
                + (0.5 * self.n4.get_noise_2d(xf * 8.0, zf * 8.0) + 0.5) / 16.0);
        if y as f32 <= h {
            if (y + 1) as f32 <= h {
                BlockId::DIRT
            } else {
                BlockId::GRASS
            }
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
    offset: [i32; 3],
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
        let mut chunk_data = [[[Block::default(); CHUNK_L]; CHUNK_H]; CHUNK_W];
        for i in 0..CHUNK_W {
            for j in 0..CHUNK_H {
                for k in 0..CHUNK_L {
                    chunk_data[i][j][k] = Block {
                        block_type: world.terrain.evaluate(
                            i as i32 + x * CHUNK_W as i32,
                            j as i32 + y * CHUNK_H as i32,
                            k as i32 + z * CHUNK_L as i32,
                        ),
                    };
                }
            }
        }

        Arc::new(Mutex::new(Self {
            world_x: x,
            world_y: y,
            world_z: z,
            chunk_data,
            chunk_mesh: None,
            world: Arc::downgrade(world),
        }))
    }

    fn is_air(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        x: usize,
        y: usize,
        z: usize,
        chunks_lock: &Result<
            MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>,
            PoisonError<MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>>,
        >,
    ) -> bool {
        if chunks_lock
            .as_ref()
            .unwrap()
            .contains_key(&(chunk_x, chunk_z))
        {
            chunks_lock
                .as_ref()
                .unwrap()
                .get(&(chunk_x, chunk_z))
                .unwrap()
                .lock()
                .unwrap()
                .chunk_data[x][y][z]
                .block_type
                == BlockId::AIR
        } else {
            false
        }
    }

    pub fn gen_mesh(
        &mut self,
        device: &wgpu::Device,
        chunks_lock: &Result<
            MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>,
            PoisonError<MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>>,
        >,
    ) {
        if self.chunk_mesh.is_some() {
            return;
        }

        let mut vertices: Vec<MCVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for i in 0..CHUNK_W {
            for j in 0..CHUNK_H {
                for k in 0..CHUNK_L {
                    let id = self.chunk_data[i][j][k].block_type;
                    let block_pos = [
                        (i as i32 + self.world_x * CHUNK_W as i32),
                        (j as i32 + self.world_y * CHUNK_H as i32),
                        (k as i32 + self.world_z * CHUNK_L as i32),
                    ];
                    if id == BlockId::AIR {
                        continue;
                    }
                    if i == 0 {
                        if self.is_air(
                            self.world_x - 1,
                            self.world_z,
                            CHUNK_W - 1,
                            j,
                            k,
                            chunks_lock,
                        ) {
                            add_face(&mut vertices, &mut indices, block_pos, id, 0);
                        }
                    } else if self.chunk_data[i - 1][j][k].block_type == BlockId::AIR {
                        add_face(&mut vertices, &mut indices, block_pos, id, 0);
                    }
                    if j == 0 || self.chunk_data[i][j - 1][k].block_type == BlockId::AIR {
                        add_face(&mut vertices, &mut indices, block_pos, id, 1);
                    }
                    if k == 0 {
                        if self.is_air(
                            self.world_x,
                            self.world_z - 1,
                            i,
                            j,
                            CHUNK_L - 1,
                            chunks_lock,
                        ) {
                            add_face(&mut vertices, &mut indices, block_pos, id, 2);
                        }
                    } else if self.chunk_data[i][j][k - 1].block_type == BlockId::AIR {
                        add_face(&mut vertices, &mut indices, block_pos, id, 2);
                    }
                    if i == CHUNK_W - 1 {
                        if self.is_air(self.world_x + 1, self.world_z, 0, j, k, chunks_lock) {
                            add_face(&mut vertices, &mut indices, block_pos, id, 3);
                        }
                    } else if self.chunk_data[i + 1][j][k].block_type == BlockId::AIR {
                        add_face(&mut vertices, &mut indices, block_pos, id, 3);
                    }
                    if j == CHUNK_H - 1 || self.chunk_data[i][j + 1][k].block_type == BlockId::AIR {
                        add_face(&mut vertices, &mut indices, block_pos, id, 4);
                    }
                    if k == CHUNK_L - 1 {
                        if self.is_air(self.world_x, self.world_z + 1, i, j, 0, chunks_lock) {
                            add_face(&mut vertices, &mut indices, block_pos, id, 5);
                        }
                    } else if self.chunk_data[i][j][k + 1].block_type == BlockId::AIR {
                        add_face(&mut vertices, &mut indices, block_pos, id, 5);
                    }
                }
            }
        }

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

const RENDER_DISTANCE: i32 = 3;
pub struct World {
    pub chunks: Mutex<HashMap<(i32, i32), Arc<Mutex<Chunk>>>>,
    material: Arc<Material>,
    terrain: Terrain,
    loaded: Mutex<HashSet<(i32, i32)>>,
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

        let terrain = Terrain::new();

        Ok(Arc::new(Self {
            chunks,
            material,
            terrain,
            loaded: Mutex::new(HashSet::new()),
        }))
    }

    pub fn gen_chunk(
        self: &Arc<Self>,
        x: i32,
        y: i32,
        z: i32,
        chunks_lock: &mut Result<
            MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>,
            PoisonError<MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>>,
        >,
    ) -> Arc<Mutex<Chunk>> {
        let chunk = Chunk::new(x, y, z, self);
        chunks_lock.as_mut().unwrap().insert((x, z), chunk.clone());
        chunk
    }

    fn load_chunk(
        self: &Arc<Self>,
        device: &wgpu::Device,
        x: i32,
        z: i32,
        chunks_lock: &Result<
            MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>,
            PoisonError<MutexGuard<'_, HashMap<(i32, i32), Arc<Mutex<Chunk>>>>>,
        >,
    ) {
        let chunk_lock = chunks_lock.as_ref().unwrap().get(&(x, z)).unwrap().lock();
        if chunk_lock.as_ref().unwrap().chunk_mesh.is_none() {
            chunk_lock.unwrap().gen_mesh(device, &chunks_lock);
        }
    }

    fn unload_chunk(&self, x: i32, z: i32) {
        self.chunks
            .lock()
            .unwrap()
            .get(&(x, z))
            .unwrap()
            .lock()
            .unwrap()
            .chunk_mesh = None;
    }

    pub fn update(self: &Arc<Self>, device: &wgpu::Device, camera: &Camera) {
        let c_x = (camera.position.x / CHUNK_W as f32).floor() as i32;
        let c_y = (camera.position.y / CHUNK_H as f32).floor() as i32;
        let c_z = (camera.position.z / CHUNK_L as f32).floor() as i32;
        let mut to_unload: Vec<(i32, i32)> = Vec::new();
        for (x, z) in self.loaded.lock().unwrap().iter() {
            let (x, z) = (*x, *z);
            if x > c_x + RENDER_DISTANCE
                || x < c_x - RENDER_DISTANCE
                || z > c_z + RENDER_DISTANCE
                || z < c_z - RENDER_DISTANCE
            {
                to_unload.push((x, z));
            }
        }
        for (x, z) in to_unload {
            self.unload_chunk(x, z);
            self.loaded.lock().unwrap().remove(&(x, z));
        }

        let mut chunks_lock = self.chunks.lock();
        for x in (c_x - RENDER_DISTANCE - 1)..(c_x + RENDER_DISTANCE + 2) {
            for z in (c_z - RENDER_DISTANCE - 1)..(c_z + RENDER_DISTANCE + 2) {
                if !chunks_lock.as_ref().unwrap().contains_key(&(x, z)) {
                    self.gen_chunk(x, 0, z, &mut chunks_lock);
                }
            }
        }

        for x in (c_x - RENDER_DISTANCE)..(c_x + RENDER_DISTANCE + 1) {
            for z in (c_z - RENDER_DISTANCE)..(c_z + RENDER_DISTANCE + 1) {
                self.load_chunk(device, x, z, &chunks_lock);
                self.loaded.lock().unwrap().insert((x, z));
            }
        }
    }

    pub fn get_model(&self) -> Model {
        let mut meshes: Vec<Weak<Mesh>> = Vec::new();
        let materials: Vec<Weak<Material>> = vec![Arc::downgrade(&self.material)];

        for k in self.loaded.lock().unwrap().iter() {
            meshes.push(Arc::downgrade(
                self.chunks
                    .lock()
                    .unwrap()
                    .get(k)
                    .unwrap()
                    .lock()
                    .unwrap()
                    .chunk_mesh
                    .as_ref()
                    .unwrap(),
            ));
        }

        Model { meshes, materials }
    }
}
