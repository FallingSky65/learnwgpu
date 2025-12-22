struct Camera {
  view_pos: vec4<f32>,
  view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
  @location(0) position: vec3<f32>,
  //@location(1) tex_coords: vec2<f32>,
  @location(1) id: u32,
}

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) tex_coords: vec2<f32>,
};

struct InstanceInput {
  @location(5) model_matrix_0: vec4<f32>,
  @location(6) model_matrix_1: vec4<f32>,
  @location(7) model_matrix_2: vec4<f32>,
  @location(8) model_matrix_3: vec4<f32>,
};

fn unpack_id(id: u32) -> vec2<u32> {
    let vertex_id: u32 = id & 0x00FF;
    let block_id: u32 = (id & 0xFF00) >> 8;
    return vec2<u32>(vertex_id, block_id);
}

const texcoord_lookup: array<vec2<f32>, 24> = array(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 0.0),

    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 0.0),

    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),

    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),

    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),

    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 0.0),
);

const texcoord_offset_lookup: array<array<vec2<f32>, 6>, 3> = array(
    array(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
    ),
    array(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    ),
    array(
        vec2<f32>(2.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(2.0, 0.0),
    ),
);

@vertex
fn vs_main(
  model: VertexInput,
  instance: InstanceInput,
  @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
  let model_matrix = mat4x4<f32>(
    instance.model_matrix_0,
    instance.model_matrix_1,
    instance.model_matrix_2,
    instance.model_matrix_3,
  );

  let unpacked_id: vec2<u32> = unpack_id(model.id);
  let vertex_id: u32 = unpacked_id.x;
  let block_id: u32 = unpacked_id.y;
  let face_id: u32 = (vertex_id >> 2);

  var out: VertexOutput;
  // out.tex_coords = model.tex_coords;
  out.tex_coords = texcoord_lookup[vertex_id];
  out.tex_coords += texcoord_offset_lookup[block_id][face_id];
  out.tex_coords.x /= 3.0;
  out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
  return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}
