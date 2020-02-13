const std = @import("std");
const builtin = @import("builtin");
const Builder = std.build.Builder;
const path = std.fs.path;

pub fn build(b: *Builder) !void {
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("main", "src/main.zig");
    exe.setBuildMode(mode);

    switch (builtin.os) {
        .windows => {
            exe.addLibPath("./libs");
            exe.linkSystemLibrary("User32");
            exe.linkSystemLibrary("Gdi32");
            exe.linkSystemLibrary("Shell32");
        },
        else => {
            @panic("Unsupported OS!");
        }
    }

    exe.linkSystemLibrary("glfw3");
    exe.linkSystemLibrary("vulkan-1");
    exe.linkSystemLibrary("c");

    const run_cmd = exe.run();
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    b.default_step.dependOn(&exe.step);
    b.installArtifact(exe);

    try addShader(b, exe, "shader.vert", "vert.spv");
    try addShader(b, exe, "shader.frag", "frag.spv");
}

fn addShader(b: *Builder, exe: var, in_file: []const u8, out_file: []const u8) !void {
    // example:
    // glslc -o shaders/vert.spv shaders/shader.vert

    const dirname = "shaders";
    const full_in = try path.join(b.allocator, &[_][]const u8{ dirname, in_file });
    const full_out = try path.join(b.allocator, &[_][]const u8{ dirname, out_file });

    // Note: glslc comes with Vulkan. Make sure that it's in your system path
    const run_cmd = b.addSystemCommand(&[_][]const u8{
        "glslc",
        "-o",
        full_out,
        full_in,
    });
    exe.step.dependOn(&run_cmd.step);
}
