// One text file containing both shader stages.
// The Python demo splits it at the two markers below.

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

    vec2 world_position = u_position + rotation * (in_position * u_size);
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

uniform sampler2D u_albedo_map;
uniform sampler2D u_normal_map;
uniform sampler2D u_metallic_map;
uniform sampler2D u_roughness_map;

uniform vec2 u_resolution;
uniform vec2 u_light_position;
uniform float u_rotation;
uniform float u_time;
uniform float u_metallic_scale;
uniform float u_roughness_scale;
uniform float u_normal_strength;
uniform float u_alpha_cutoff;
uniform int u_debug_view;

in vec2 v_uv;
in vec2 v_world_position;

out vec4 frag_color;

const float PI = 3.141592653589793;

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

float distribution_ggx(vec3 normal, vec3 halfway, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(normal, halfway), 0.0);
    float n_dot_h2 = n_dot_h * n_dot_h;
    float denominator = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 0.000001);
}

float geometry_schlick_ggx(float n_dot_x, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_x / max(n_dot_x * (1.0 - k) + k, 0.000001);
}

float geometry_smith(
    vec3 normal,
    vec3 view_direction,
    vec3 light_direction,
    float roughness
) {
    float n_dot_v = max(dot(normal, view_direction), 0.0);
    float n_dot_l = max(dot(normal, light_direction), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness)
         * geometry_schlick_ggx(n_dot_l, roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - saturate(cos_theta), 5.0);
}

vec3 fresnel_schlick_roughness(
    float cos_theta,
    vec3 f0,
    float roughness
) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0)
        * pow(1.0 - saturate(cos_theta), 5.0);
}

vec3 sample_world_normal(vec2 uv) {
    vec3 tangent_normal = texture(u_normal_map, uv).rgb * 2.0 - 1.0;
    tangent_normal.xy *= u_normal_strength;
    tangent_normal = normalize(tangent_normal);

    // Normal-map green is tangent-space up. Pygame screen Y points down,
    // so invert it before rotating the tangent normal into screen space.
    vec2 screen_xy = vec2(tangent_normal.x, -tangent_normal.y);

    float c = cos(u_rotation);
    float s = sin(u_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    screen_xy = rotation * screen_xy;

    return normalize(vec3(screen_xy, tangent_normal.z));
}

vec3 procedural_environment(vec3 reflection_direction, float roughness) {
    // Screen-space negative Y is upward toward the sky.
    float sky_amount = saturate(-reflection_direction.y * 0.5 + 0.5);

    vec3 ground = vec3(0.11, 0.045, 0.018);
    vec3 horizon = vec3(1.10, 0.44, 0.12);
    vec3 sky = vec3(0.055, 0.28, 0.86);
    vec3 zenith = vec3(0.008, 0.020, 0.075);

    vec3 lower = mix(ground, horizon, smoothstep(0.0, 0.50, sky_amount));
    vec3 upper = mix(sky, zenith, smoothstep(0.50, 1.0, sky_amount));
    vec3 environment = mix(lower, upper, smoothstep(0.43, 0.57, sky_amount));

    vec3 highlight_direction = normalize(vec3(
        0.55 + sin(u_time * 0.45) * 0.16,
        -0.46,
        0.70
    ));
    float highlight_power = mix(240.0, 7.0, roughness);
    float highlight = pow(
        max(dot(reflection_direction, highlight_direction), 0.0),
        highlight_power
    );
    environment += vec3(7.0, 5.2, 3.2) * highlight;

    vec3 average_environment = vec3(0.22, 0.24, 0.29);
    return mix(
        environment,
        average_environment,
        roughness * roughness * 0.82
    );
}

vec3 aces_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp(
        (color * (a * color + b)) /
        (color * (c * color + d) + e),
        0.0,
        1.0
    );
}

void main() {
    vec4 albedo_sample = texture(u_albedo_map, v_uv);
    if (albedo_sample.a < u_alpha_cutoff) {
        discard;
    }

    vec3 normal_sample = texture(u_normal_map, v_uv).rgb;
    float metallic_sample = texture(u_metallic_map, v_uv).r;
    float roughness_sample = texture(u_roughness_map, v_uv).r;

    // Debug views display the actual source maps loaded by Python.
    if (u_debug_view == 1) {
        frag_color = vec4(albedo_sample.rgb, albedo_sample.a);
        return;
    }
    if (u_debug_view == 2) {
        frag_color = vec4(normal_sample, albedo_sample.a);
        return;
    }
    if (u_debug_view == 3) {
        frag_color = vec4(vec3(metallic_sample), albedo_sample.a);
        return;
    }
    if (u_debug_view == 4) {
        frag_color = vec4(vec3(roughness_sample), albedo_sample.a);
        return;
    }

    vec3 albedo = pow(albedo_sample.rgb, vec3(2.2));
    float metallic = saturate(metallic_sample * u_metallic_scale);
    float roughness = clamp(
        roughness_sample * u_roughness_scale,
        0.045,
        1.0
    );

    vec3 normal = sample_world_normal(v_uv);
    vec3 view_direction = vec3(0.0, 0.0, 1.0);

    vec2 light_offset = u_light_position - v_world_position;
    float scene_scale = max(min(u_resolution.x, u_resolution.y), 1.0);
    vec3 light_vector = vec3(light_offset / scene_scale, 0.32);
    float light_distance = max(length(light_vector), 0.001);
    vec3 light_direction = light_vector / light_distance;
    vec3 halfway = normalize(view_direction + light_direction);

    float attenuation = 1.0 / (0.11 + light_distance * light_distance * 2.35);
    vec3 radiance = vec3(9.0, 7.2, 5.8) * attenuation;

    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    float n_dot_v = max(dot(normal, view_direction), 0.0);
    float n_dot_l = max(dot(normal, light_direction), 0.0);

    float distribution = distribution_ggx(normal, halfway, roughness);
    float geometry = geometry_smith(
        normal,
        view_direction,
        light_direction,
        roughness
    );
    vec3 fresnel = fresnel_schlick(
        max(dot(halfway, view_direction), 0.0),
        f0
    );

    vec3 numerator = distribution * geometry * fresnel;
    float denominator = max(4.0 * n_dot_v * n_dot_l, 0.0001);
    vec3 specular = numerator / denominator;

    vec3 k_specular = fresnel;
    vec3 k_diffuse = (vec3(1.0) - k_specular) * (1.0 - metallic);
    vec3 direct_light = (
        k_diffuse * albedo / PI + specular
    ) * radiance * n_dot_l;

    vec3 reflection_direction = reflect(-view_direction, normal);
    vec3 environment = procedural_environment(reflection_direction, roughness);
    vec3 environment_fresnel = fresnel_schlick_roughness(
        n_dot_v,
        f0,
        roughness
    );

    vec3 ambient_diffuse = (
        albedo * vec3(0.045, 0.055, 0.075) * (1.0 - metallic)
    );
    vec3 ambient_specular = environment * environment_fresnel;

    vec3 color = direct_light + ambient_diffuse + ambient_specular;
    color = aces_tonemap(color);
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, albedo_sample.a);
}
