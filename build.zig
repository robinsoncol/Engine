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
            exe.linkSystemLibrary("./platforms/windows/libs/glfw3");
            exe.linkSystemLibrary("./platforms/windows/libs/vulkan-1");
            exe.linkSystemLibrary("User32");
            exe.linkSystemLibrary("Gdi32");
            exe.linkSystemLibrary("Shell32");
        },
        else => {
            @panic("Unsupported OS!");
        }
    }
    exe.linkSystemLibrary("c");

    const runCmd = exe.run();
    const runStep = b.step("run", "Run the app");
    runStep.dependOn(&runCmd.step);

    b.default_step.dependOn(&exe.step);
    b.installArtifact(exe);

    try addShader(b, exe, "shader.vert", "vert.spv");
    try addShader(b, exe, "shader.frag", "frag.spv");
}

fn addShader(b: *Builder, exe: var, in_file: []const u8, out_file: []const u8) !void {
    // example:
    // glslc -o shaders/vert.spv shaders/shader.vert

    const dirname = "shaders";
    const fullIn = try path.join(b.allocator, &[_][]const u8{ dirname, in_file });
    const fullOut = try path.join(b.allocator, &[_][]const u8{ dirname, out_file });

    const runCmd = b.addSystemCommand(&[_][]const u8{
        "platforms/windows/tools/glslc.exe",
        "-o",
        fullOut,
        fullIn,
    });
    exe.step.dependOn(&runCmd.step);
}