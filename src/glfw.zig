const v = @import("vulkan_core.zig");

pub const GLFW_CLIENT_API = 139265;
pub const GLFW_NO_API = 0;
pub const GLFW_RESIZABLE = 131075;
pub const GLFW_FALSE = 0;

pub const GLFWmonitor = @OpaqueType();
pub const GLFWwindow = @OpaqueType();

pub extern fn glfwInit() u32;
pub extern fn glfwTerminate() void;
pub extern fn glfwWindowHint(hint: u32, value: u32) void;
pub extern fn glfwCreateWindow(width: u32, height: u32, title: ?[*]const u8, monitor: ?*GLFWmonitor, share: ?*GLFWwindow) ?*GLFWwindow;
pub extern fn glfwGetRequiredInstanceExtensions(count: *u32) [*]const [*]const u8;
pub extern fn glfwCreateWindowSurface(instance: v.Instance, window: *GLFWwindow, allocator: ?*const v.AllocationCallbacks, surface: *v.SurfaceKHR) v.Result;
pub extern fn glfwDestroyWindow(window: ?*GLFWwindow) void;
pub extern fn glfwWindowShouldClose(window: ?*GLFWwindow) u32;
pub extern fn glfwPollEvents() void;