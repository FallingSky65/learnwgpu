use std::{
    io::{BufReader, Cursor},
    sync::{Arc, Weak},
};

use gltf::Gltf;
use image::{DynamicImage, RgbaImage, load_from_memory_with_format};
use wgpu::util::DeviceExt;

use crate::{
    model::{self, Material, ModelVertex},
    texture::{self, Texture},
};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("res") {
        origin = format!("{}/res", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    #[cfg(target_arch = "wasm32")]
    let txt = {
        let url = format_url(file_name);
        reqwest::get(url).await?.text().await?
    };
    #[cfg(not(target_arch = "wasm32"))]
    let txt = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read_to_string(path)?
    };

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    let data = {
        let url = format_url(file_name);
        reqwest::get(url).await?.bytes().await?.to_vec()
    };
    #[cfg(not(target_arch = "wasm32"))]
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_obj(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials: Vec<Weak<Material>> = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture.unwrap(), device, queue).await?;
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

        let material = Arc::new(model::Material {
            name: m.name,
            diffuse_texture,
            bind_group,
        });

        materials.push(Arc::downgrade(&material))
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| {
                    if m.mesh.normals.is_empty() {
                        model::ModelVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ],
                            tex_coords: [
                                m.mesh.texcoords[i * 2],
                                1.0 - m.mesh.texcoords[i * 2 + 1],
                            ],
                            normal: [0.0, 0.0, 0.0],
                        }
                    } else {
                        model::ModelVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ],
                            tex_coords: [
                                m.mesh.texcoords[i * 2],
                                1.0 - m.mesh.texcoords[i * 2 + 1],
                            ],
                            normal: [
                                m.mesh.normals[i * 3],
                                m.mesh.normals[i * 3 + 1],
                                m.mesh.normals[i * 3 + 2],
                            ],
                        }
                    }
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            let mesh = Arc::new(model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer: index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            });

            Arc::downgrade(&mesh)
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}

pub async fn load_glb(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let glb_data = load_binary(file_name).await.unwrap();
    let (gltf, buffers, images) = gltf::import_slice(glb_data.as_slice()).unwrap();

    println!("gltf has {} meshes", gltf.meshes().len());
    println!("gltf has {} buffers", gltf.buffers().len());
    println!("gltf has {} materials", gltf.materials().len());
    println!("gltf has {} textures", gltf.textures().len());
    println!("gltf has {} images", gltf.images().len());

    let mut materials: Vec<Weak<Material>> = Vec::new();

    for m in gltf.materials() {
        let texture_info = m.pbr_metallic_roughness().base_color_texture().unwrap();
        let image = texture_info.texture().source();

        match image.source() {
            gltf::image::Source::Uri { uri, mime_type } => {
                println!("Image URI: {}", uri);
                let diffuse_texture = load_texture(uri, device, queue).await.unwrap();
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
                    name: m.name().unwrap().to_string(),
                    diffuse_texture,
                    bind_group,
                });

                materials.push(Arc::downgrade(&material));
            }
            gltf::image::Source::View { view, mime_type } => {
                println!("Embedded image MIME type: {}", mime_type);
                let image_data = &images[image.index()];
                println!("Image Dims: {}, {}", image_data.width, image_data.height);
                println!("Image Format: {:?}", image_data.format);
                let image_rgba = DynamicImage::ImageRgba8(
                    RgbaImage::from_raw(
                        image_data.width,
                        image_data.height,
                        image_data.pixels.clone(),
                    )
                    .unwrap(),
                );
                let diffuse_texture =
                    Texture::from_image(device, queue, &image_rgba, texture_info.texture().name())
                        .unwrap();
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
                    name: m.name().unwrap().to_string(),
                    diffuse_texture,
                    bind_group,
                });
                materials.push(Arc::downgrade(&material));
            }
        }
    }

    let meshes = gltf
        .meshes()
        .map(|m| {
            let p = m.primitives().next().unwrap();

            let mut vertices: Vec<ModelVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();

            let r = p.reader(|buffer| Some(&buffers[buffer.index()]));

            if let Some(gltf::mesh::util::ReadIndices::U16(gltf::accessor::Iter::Standard(iter))) =
                r.read_indices()
            {
                for v in iter {
                    indices.push(v as u32);
                }
            }
            if let Some(gltf::mesh::util::ReadIndices::U32(gltf::accessor::Iter::Standard(iter))) =
                r.read_indices()
            {
                for v in iter {
                    indices.push(v);
                }
            }

            if let (
                Some(positions_iter),
                Some(gltf::mesh::util::ReadTexCoords::F32(gltf::accessor::Iter::Standard(
                    texcoords_iter,
                ))),
                Some(normals_iter),
            ) = (r.read_positions(), r.read_tex_coords(0), r.read_normals())
            {
                let zipped = positions_iter.zip(texcoords_iter).zip(normals_iter);
                for ((position, texcoord), normal) in zipped {
                    vertices.push(ModelVertex {
                        position,
                        tex_coords: texcoord,
                        normal,
                    });
                }
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            let material = p.material();
            let material_index = if material.index().is_some() {
                material.index().unwrap()
            } else {
                0
            };

            let mesh = Arc::new(model::Mesh {
                name: m.name().unwrap().to_string(),
                vertex_buffer: vertex_buffer,
                index_buffer: index_buffer,
                num_elements: if indices.len() == 0 {
                    vertices.len()
                } else {
                    indices.len()
                } as u32,
                material: material_index as usize,
            });

            Arc::downgrade(&mesh)
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}
