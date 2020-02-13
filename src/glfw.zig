const v = @import("vulkan_core.zig");

pub const GLFW_CLIENT_API = 139265;
pub const GLFW_NO_API = 0;
pub const GLFW_RESIZABLE = 131075;
pub const GLFW_FALSE = 0;

pub const GLFWmonitor = @OpaqueType();
pub const GLFWwindow = @OpaqueType();

pub extern fn glfwInit() c_int;
pub extern fn glfwTerminate() void;
pub extern fn glfwWindowHint(hint: c_int, value: c_int) void;
pub extern fn glfwCreateWindow(width: c_int, height: c_int, title: ?[*]const u8, monitor: ?*GLFWmonitor, share: ?*GLFWwindow) ?*GLFWwindow;
pub extern fn glfwGetRequiredInstanceExtensions(count: *u32) [*]const [*]const u8;
pub extern fn glfwCreateWindowSurface(instance: v.Instance, window: *GLFWwindow, allocator: ?*const v.AllocationCallbacks, surface: *v.SurfaceKHR) v.Result;
pub extern fn glfwDestroyWindow(window: ?*GLFWwindow) void;
pub extern fn glfwWindowShouldClose(window: ?*GLFWwindow) c_int;
pub extern fn glfwPollEvents() void;