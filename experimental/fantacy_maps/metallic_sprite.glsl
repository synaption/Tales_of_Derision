// One file containing both shader stages.
// The Python demo splits the file at the markers below.

// === VERTEX SHADER ===
#version 330

in vec2 in_position;
in vec2 in_uv;

uniform vec2 u_resolution;
uniform vec2 u_position;
uniform vec2 u_size;
uniform float u_rotation;

out vec2 v_uv;
out vec2 v_world_position;

void main() {
    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);

    // Sprite coordinates use Pygame's top-left origin and downward-positive Y.
    vec2 world_position = u_position + rotation * (in_position * u_size);

    // Convert pixel coordinates to OpenGL normalized device coordinates.
    vec2 ndc = vec2(
        world_position.x / u_resolution.x * 2.0 - 1.0,
        1.0 - world_position.y / u_resolution.y * 2.0
    );

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_world_position = world_position;
}

// === FRAGMENT SHADER ===
#version 330

uniform sampler2D u_sprite;
uniform vec2 u_resolution;
uniform vec2 u_light_position;
uniform vec2 u_texel_size;
uniform float u_time;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_normal_strength;
uniform float u_alpha_cutoff;

in vec2 v_uv;
in vec2 v_world_position;

out vec4 frag_color;

const float PI = 3.141592653589793;

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float distribution_ggx(vec3 normal, vec3 halfway, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(normal, halfway), 0.0);
    float n_dot_h2 = n_dot_h * n_dot_h;

    float denominator = n_dot_h2 * (a2 - 1.0) + 1.0;
    denominator = PI * denominator * denominator;
    return a2 / max(denominator, 0.000001);
}

float geometry_schlick_ggx(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.000001);
}

float geometry_smith(vec3 normal, vec3 view_direction, vec3 light_direction, float roughness) {
    float n_dot_v = max(dot(normal, view_direction), 0.0);
    float n_dot_l = max(dot(normal, light_direction), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness)
         * geometry_schlick_ggx(n_dot_l, roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - saturate(cos_theta), 5.0);
}

vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0)
        * pow(1.0 - saturate(cos_theta), 5.0);
}

float sprite_height(vec2 uv) {
    vec4 texel = texture(u_sprite, clamp(uv, vec2(0.0), vec2(1.0)));

    // The image's brightness acts as a lightweight height map. Alpha keeps
    // transparent pixels from affecting the generated normal at the border.
    return luminance(pow(texel.rgb, vec3(2.2))) * texel.a;
}

vec3 generated_normal(vec2 uv) {
    float left_height = sprite_height(uv - vec2(u_texel_size.x, 0.0));
    float right_height = sprite_height(uv + vec2(u_texel_size.x, 0.0));
    float down_height = sprite_height(uv - vec2(0.0, u_texel_size.y));
    float up_height = sprite_height(uv + vec2(0.0, u_texel_size.y));

    float dx = right_height - left_height;
    float dy = up_height - down_height;

    return normalize(vec3(-dx * u_normal_strength,
                          -dy * u_normal_strength,
                          1.0));
}

vec3 procedural_environment(vec3 reflection_direction, float roughness) {
    float sky_amount = saturate(reflection_direction.y * 0.5 + 0.5);

    vec3 ground = vec3(0.12, 0.055, 0.025);
    vec3 horizon = vec3(0.95, 0.46, 0.16);
    vec3 sky = vec3(0.08, 0.35, 0.90);
    vec3 zenith = vec3(0.015, 0.035, 0.11);

    vec3 lower = mix(ground, horizon, smoothstep(0.0, 0.52, sky_amount));
    vec3 upper = mix(sky, zenith, smoothstep(0.52, 1.0, sky_amount));
    vec3 environment = mix(lower, upper, smoothstep(0.42, 0.58, sky_amount));

    // A moving bright source makes the reflection visibly respond over time.
    vec3 sun_direction = normalize(vec3(
        0.55 + sin(u_time * 0.45) * 0.18,
        0.48,
        0.68
    ));
    float sun_exponent = mix(220.0, 8.0, roughness);
    float sun = pow(max(dot(reflection_direction, sun_direction), 0.0), sun_exponent);
    environment += vec3(6.0, 4.8, 3.2) * sun;

    // Roughness approximates a blurred reflection by pulling toward the
    // environment's average color.
    vec3 average_environment = vec3(0.24, 0.25, 0.29);
    return mix(environment, average_environment, roughness * roughness * 0.78);
}

vec3 aces_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) /
                 (color * (c * color + d) + e), 0.0, 1.0);
}

void main() {
    vec4 sprite_sample = texture(u_sprite, v_uv);
    if (sprite_sample.a < u_alpha_cutoff) {
        discard;
    }

    vec3 albedo = pow(sprite_sample.rgb, vec3(2.2));
    float metallic = saturate(u_metallic);
    float roughness = clamp(u_roughness, 0.045, 1.0);

    vec3 normal = generated_normal(v_uv);
    vec3 view_direction = vec3(0.0, 0.0, 1.0);

    vec2 light_offset = u_light_position - v_world_position;
    float scene_scale = max(min(u_resolution.x, u_resolution.y), 1.0);
    vec3 light_vector = vec3(light_offset / scene_scale, 0.30);
    float light_distance = max(length(light_vector), 0.001);
    vec3 light_direction = light_vector / light_distance;
    vec3 halfway = normalize(view_direction + light_direction);

    // Mouse-controlled HDR point light.
    float attenuation = 1.0 / (0.10 + light_distance * light_distance * 2.2);
    vec3 radiance = vec3(9.0, 7.2, 5.8) * attenuation;

    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    float n_dot_v = max(dot(normal, view_direction), 0.0);
    float n_dot_l = max(dot(normal, light_direction), 0.0);

    float distribution = distribution_ggx(normal, halfway, roughness);
    float geometry = geometry_smith(normal, view_direction, light_direction, roughness);
    vec3 fresnel = fresnel_schlick(max(dot(halfway, view_direction), 0.0), f0);

    vec3 numerator = distribution * geometry * fresnel;
    float denominator = max(4.0 * n_dot_v * n_dot_l, 0.0001);
    vec3 specular = numerator / denominator;

    vec3 k_specular = fresnel;
    vec3 k_diffuse = (vec3(1.0) - k_specular) * (1.0 - metallic);
    vec3 direct_light = (k_diffuse * albedo / PI + specular) * radiance * n_dot_l;

    vec3 reflection_direction = reflect(-view_direction, normal);
    vec3 environment = procedural_environment(reflection_direction, roughness);
    vec3 environment_fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    vec3 ambient_diffuse = albedo * vec3(0.055, 0.065, 0.085) * (1.0 - metallic);
    vec3 ambient_specular = environment * environment_fresnel;

    vec3 color = direct_light + ambient_diffuse + ambient_specular;
    color = aces_tonemap(color);
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, sprite_sample.a);
}
