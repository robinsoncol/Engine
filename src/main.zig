const std = @import("std");
const Allocator = std.mem.Allocator;
const maxInt = std.math.maxInt;

const vk = @import("./bindings/vulkan.zig");
const glfw = @import("./bindings/glfw.zig");

const WIDTH = 1280;
const HEIGHT = 720;

const MAX_FRAMES_IN_FLIGHT = 2;

const enableValidationLayers = std.debug.runtime_safety;
const validationLayers = [_][*:0]const u8{"VK_LAYER_LUNARG_standard_validation"};
const deviceExtensions = [_][*:0]const u8{vk.KHR_SWAPCHAIN_EXTENSION_NAME};

var currentFrame: usize = 0;
var instance: vk.Instance = undefined;
var callback: vk.DebugReportCallbackEXT = undefined;
var surface: vk.SurfaceKHR = undefined;
var physicalDevice: vk.PhysicalDevice = undefined;
var globalDevice: vk.Device = undefined;
var graphicsQueue: vk.Queue = undefined;
var presentQueue: vk.Queue = undefined;
var swapChainImages: []vk.Image = undefined;
var swapChain: vk.SwapchainKHR = undefined;
var swapChainImageFormat: vk.Format = undefined;
var swapChainExtent: vk.Extent2D = undefined;
var swapChainImageViews: []vk.ImageView = undefined;
var renderPass: vk.RenderPass = undefined;
var pipelineLayout: vk.PipelineLayout = undefined;
var graphicsPipeline: vk.Pipeline = undefined;
var swapChainFramebuffers: []vk.Framebuffer = undefined;
var commandPool: vk.CommandPool = undefined;
var commandBuffers: []vk.CommandBuffer = undefined;
var vertexBuffer: vk.Buffer = undefined;
var vertexBufferMemory:  vk.DeviceMemory = undefined;

var imageAvailableSemaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = undefined;
var renderFinishedSemaphores: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = undefined;
var inFlightFences: [MAX_FRAMES_IN_FLIGHT]vk.Fence = undefined;

const QueueFamilyIndices = struct {
    graphicsFamily: ?u32,
    presentFamily: ?u32,

    fn init() QueueFamilyIndices {
        return QueueFamilyIndices{
            .graphicsFamily = null,
            .presentFamily = null,
        };
    }

    fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphicsFamily != null and self.presentFamily != null;
    }
};

const SwapChainSupportDetails = struct {
    capabilities: vk.SurfaceCapabilitiesKHR,
    formats: std.ArrayList(vk.SurfaceFormatKHR),
    presentModes: std.ArrayList(vk.PresentModeKHR),

    fn init(allocator: *Allocator) SwapChainSupportDetails {
        var result = SwapChainSupportDetails{
            .capabilities = undefined,
            .formats = std.ArrayList(vk.SurfaceFormatKHR).init(allocator),
            .presentModes = std.ArrayList(vk.PresentModeKHR).init(allocator),
        };
        const slice = std.mem.sliceAsBytes(@as(*[1]vk.SurfaceCapabilitiesKHR, &result.capabilities)[0..1]);
        std.mem.set(u8, slice, 0);
        return result;
    }

    fn deinit(self: *SwapChainSupportDetails) void {
        self.formats.deinit();
        self.presentModes.deinit();
    }
};

const Vertex = packed struct {
    // Attributes to be passed to the vertex shader
    position: [2]f32,
    color: [3]f32,

    pub fn getBindingDescriptions() [1]vk.VertexInputBindingDescription {
        return [1]vk.VertexInputBindingDescription{
            vk.VertexInputBindingDescription{
                .binding = 0,
                .stride = @sizeOf(Vertex),
                .inputRate = vk.VertexInputRate.VERTEX
            }
        };
    }

    // An attribute description struct describes how to extract a vertex attribute from a chunk of vertex data originating from a binding description
    pub fn getAttributeDescriptions() [2]vk.VertexInputAttributeDescription {
        return [2]vk.VertexInputAttributeDescription{
            vk.VertexInputAttributeDescription{
                .binding = 0,
                // References the location in the vertex shader
                .location = 0,
                // Oddly, we use color formats to describe an attribute
                .format = vk.Format.R32G32_SFLOAT,
                .offset = @byteOffsetOf(Vertex, "position")
            },
            vk.VertexInputAttributeDescription{
                .binding = 0,
                .location = 1,
                .format = vk.Format.R32G32B32_SFLOAT,
                .offset = @byteOffsetOf(Vertex, "color")
            }
        };
    }
};

const vertices = [_]Vertex{
    Vertex {
        .position = [_]f32{ 0.0, -0.5 },
        .color = [_]f32{ 1.0, 0.0, 0.0 }
    },
    Vertex {
        .position = [_]f32{ 0.5, 0.5},
        .color = [_]f32{ 0.0, 1.0, 0.0 }
    },
    Vertex {
        .position = [_]f32{ -0.5, 0.5 },
        .color = [_]f32{ 0.0, 0.0, 1.0 }
    }
};

pub fn main() !void {
    if (glfw.glfwInit() == 0) return error.GlfwInitFailed;
    defer glfw.glfwTerminate();

    glfw.glfwWindowHint(glfw.GLFW_CLIENT_API, glfw.GLFW_NO_API);
    glfw.glfwWindowHint(glfw.GLFW_RESIZABLE, glfw.GLFW_FALSE);

    const window = glfw.glfwCreateWindow(WIDTH, HEIGHT, "Zig Vulkan Triangle", null, null) orelse return error.GlfwCreateWindowFailed;
    defer glfw.glfwDestroyWindow(window);

    const allocator = std.heap.c_allocator;
    try initVulkan(allocator, window);

    while (glfw.glfwWindowShouldClose(window) == 0) {
        glfw.glfwPollEvents();
        try drawFrame();
    }
    try checkSuccess(vk.vkDeviceWaitIdle(globalDevice));

    cleanup();
}

fn cleanup() void {
    var i: usize = 0;
    while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
        vk.vkDestroySemaphore(globalDevice, renderFinishedSemaphores[i], null);
        vk.vkDestroySemaphore(globalDevice, imageAvailableSemaphores[i], null);
        vk.vkDestroyFence(globalDevice, inFlightFences[i], null);
    }

    vk.vkDestroyCommandPool(globalDevice, commandPool, null);

    for (swapChainFramebuffers) |framebuffer| {
        vk.vkDestroyFramebuffer(globalDevice, framebuffer, null);
    }

    vk.vkDestroyPipeline(globalDevice, graphicsPipeline, null);
    vk.vkDestroyPipelineLayout(globalDevice, pipelineLayout, null);
    vk.vkDestroyRenderPass(globalDevice, renderPass, null);

    for (swapChainImageViews) |imageView| {
        vk.vkDestroyImageView(globalDevice, imageView, null);
    }

    vk.vkDestroySwapchainKHR(globalDevice, swapChain, null);

    vk.vkDestroyBuffer(globalDevice, vertexBuffer, null);
    vk.vkFreeMemory(globalDevice, vertexBufferMemory, null);

    vk.vkDestroyDevice(globalDevice, null);

    if (enableValidationLayers) {
        DestroyDebugReportCallbackEXT(null);
    }

    vk.vkDestroySurfaceKHR(instance, surface, null);
    vk.vkDestroyInstance(instance, null);
}

fn initVulkan(allocator: *Allocator, window: *glfw.GLFWwindow) !void {
    try createInstance(allocator);
    try setupDebugCallback();
    try createSurface(window);
    try pickPhysicalDevice(allocator);
    try createLogicalDevice(allocator);

    // A swap chain is a queue of images that are waiting to be presented to the screen
    try createSwapChain(allocator);

    try createImageViews(allocator);
    try createRenderPass();
    try createGraphicsPipeline(allocator);

    // A framebuffer (frame buffer, or sometimes framestore) is a portion of random-access memory (RAM) containing a bitmap
    // that drives a video display. It is a memory buffer containing a complete frame of data. Modern video cards contain
    // framebuffer circuitry in their cores. This circuitry converts an in-memory bitmap into a video signal that can be
    // displayed on a computer monitor.
    try createFramebuffers(allocator);
    try createCommandPool(allocator);
    try createVertexBuffer(allocator);
    try createCommandBuffers(allocator);
    try createSyncObjects();
}

fn createVertexBuffer(allocator: *Allocator) !void {
    const bufferInfo = vk.BufferCreateInfo{
        .sType = vk.StructureType.BUFFER_CREATE_INFO,
        .size = @sizeOf(Vertex) * vertices.len,
        .usage = vk.BufferUsageFlagBits.VERTEX_BUFFER_BIT,
        .sharingMode = vk.SharingMode.EXCLUSIVE
    };

    try checkSuccess(vk.vkCreateBuffer(globalDevice, &bufferInfo, null, &vertexBuffer));

    var memoryRequirements : vk.MemoryRequirements = undefined;
    vk.vkGetBufferMemoryRequirements(globalDevice, vertexBuffer, &memoryRequirements);

    const allocInfo = vk.MemoryAllocateInfo{
        .sType = vk.StructureType.MEMORY_ALLOCATE_INFO,
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex = try findMemoryType(memoryRequirements.memoryTypeBits, vk.MemoryPropertyFlagBits.HOST_VISIBLE_BIT | vk.MemoryPropertyFlagBits.HOST_COHERENT_BIT)
    };

    try checkSuccess(vk.vkAllocateMemory(globalDevice, &allocInfo, null, &vertexBufferMemory));

    try checkSuccess(vk.vkBindBufferMemory(globalDevice, vertexBuffer, vertexBufferMemory, 0));

    var data : *c_void = undefined;
    try checkSuccess(vk.vkMapMemory(globalDevice, vertexBufferMemory, 0, bufferInfo.size, 0, &data));
    @memcpy(@ptrCast([*]u8, data), @ptrCast([*] const u8, &vertices), bufferInfo.size);
    vk.vkUnmapMemory(globalDevice, vertexBufferMemory);
}

fn findMemoryType(typeFilter: u32, properties: vk.FormatFeatureFlags) !u32 {
    var memoryProperties: vk.PhysicalDeviceMemoryProperties = undefined;
    vk.vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    var i: u32 = 0;
    while (i < memoryProperties.memoryTypeCount) : (i += 1) {
       const value = @floatToInt(u32, @exp2(@intToFloat(f32, i)));
        if ((typeFilter & value) != 0 and (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    } else return error.FailedToFindSuitableMemoryType;
}

fn createInstance(allocator: *Allocator) !void {
    if (enableValidationLayers) {
        if (!(try checkValidationLayerSupport(allocator))) {
            return error.ValidationLayerRequestedButNotAvailable;
        }
    }

    const appInfo = vk.ApplicationInfo{
        .sType = vk.StructureType.APPLICATION_INFO,
        .apiVersion = vk.MAKE_VERSION(1, 2, 0),
        .applicationVersion = vk.MAKE_VERSION(1, 0, 0),
        .engineVersion = vk.MAKE_VERSION(1, 0, 0),
        .pApplicationName = null,
        .pEngineName = null,
        .pNext = null,
    };

    const extensions = try getRequiredExtensions(allocator);
    defer allocator.free(extensions);

    const createInfo = vk.InstanceCreateInfo{
        .sType = vk.StructureType.INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,

        .enabledExtensionCount = @intCast(u32, extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,

        // Validation layers that we want to enable
        .enabledLayerCount = if (enableValidationLayers) @intCast(u32, validationLayers.len) else 0,
        .ppEnabledLayerNames = if (enableValidationLayers) &validationLayers else null,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(vk.vkCreateInstance(&createInfo, null, &instance));
}

/// Caller must free returned memory
fn getRequiredExtensions(allocator: *Allocator) ![][*]const u8 {
    var glfwExtensionCount: u32 = 0;
    var glfwExtensions: [*]const [*]const u8 = glfw.glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    var extensions = std.ArrayList([*]const u8).init(allocator);
    errdefer extensions.deinit();

    try extensions.appendSlice(glfwExtensions[0..glfwExtensionCount]);

    if (enableValidationLayers) {
        // TODO: Got this from the vulkan tutorial. Find out if we need this and/or
        // if this is better/worse than VK_EXT_DEBUG_REPORT_EXTENSION_NAME.
        // try extensions.append(vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        try extensions.append(vk.EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions.toOwnedSlice();
}

fn createSurface(window: *glfw.GLFWwindow) !void {
    if (glfw.glfwCreateWindowSurface(instance, window, null, &surface) != vk.Result.SUCCESS) {
        return error.FailedToCreateWindowSurface;
    }
}

fn pickPhysicalDevice(allocator: *Allocator) !void {
    var deviceCount: u32 = 0;

    // Note: The instance has the required api version, which might limit the available device(s)
    try checkSuccess(vk.vkEnumeratePhysicalDevices(instance, &deviceCount, null));

    if (deviceCount == 0) {
        return error.FailedToFindGPUsWithVulkanSupport;
    }

    const devices = try allocator.alloc(vk.PhysicalDevice, deviceCount);
    defer allocator.free(devices);
    try checkSuccess(vk.vkEnumeratePhysicalDevices(instance, &deviceCount, devices.ptr));

    physicalDevice = for (devices) |device| {
        if (try isDeviceSuitable(allocator, device)) {
            break device;
        }
    } else return error.FailedToFindSuitableGPU;
}

fn isDeviceSuitable(allocator: *Allocator, device: vk.PhysicalDevice) !bool {
    const indices = try findQueueFamilies(allocator, device);

    const extensionsSupported = try checkDeviceExtensionSupport(allocator, device);

    var swapChainAdequate = false;
    if (extensionsSupported) {
        var swapChainSupport = try querySwapChainSupport(allocator, device);
        defer swapChainSupport.deinit();
        swapChainAdequate = swapChainSupport.formats.items.len != 0 and swapChainSupport.presentModes.items.len != 0;
    }

    return indices.isComplete() and extensionsSupported and swapChainAdequate;
}

// All operations are processed in a queue. We return the first queue family that meets our needs.
fn findQueueFamilies(allocator: *Allocator, device: vk.PhysicalDevice) !QueueFamilyIndices {
    var indices = QueueFamilyIndices.init();

    var queueFamilyCount: u32 = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);

    const queueFamilies = try allocator.alloc(vk.QueueFamilyProperties, queueFamilyCount);
    defer allocator.free(queueFamilies);
    vk.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.ptr);

    var i: u32 = 0;
    for (queueFamilies) |queueFamily| {
        if (queueFamily.queueCount > 0 and
            queueFamily.queueFlags & vk.QueueFlagBits.GRAPHICS_BIT != 0)
        {
            indices.graphicsFamily = i;
        }

        var presentSupport: vk.Bool32 = 0;
        try checkSuccess(vk.vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport));

        if (queueFamily.queueCount > 0 and presentSupport != 0) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i += 1;
    }

    return indices;
}

fn checkDeviceExtensionSupport(allocator: *Allocator, device: vk.PhysicalDevice) !bool {
    var extensionCount: u32 = undefined;
    try checkSuccess(vk.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, null));

    const availableExtensions = try allocator.alloc(vk.ExtensionProperties, extensionCount);
    defer allocator.free(availableExtensions);
    try checkSuccess(vk.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, availableExtensions.ptr));

    var requiredExtensions = std.HashMap([*]const u8, void, hash_cstr, eql_cstr).init(allocator);
    defer requiredExtensions.deinit();
    for (deviceExtensions) |device_ext| {
        _ = try requiredExtensions.put(device_ext, {});
    }

    for (availableExtensions) |extension| {
        _ = requiredExtensions.remove(&extension.extensionName);
    }

    return requiredExtensions.count() == 0;
}

fn createCommandBuffers(allocator: *Allocator) !void {
    commandBuffers = try allocator.alloc(vk.CommandBuffer, swapChainFramebuffers.len);

    const allocInfo = vk.CommandBufferAllocateInfo{
        .sType = vk.StructureType.COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = vk.CommandBufferLevel.PRIMARY,
        .commandBufferCount = @intCast(u32, commandBuffers.len),
        .pNext = null,
    };

    try checkSuccess(vk.vkAllocateCommandBuffers(globalDevice, &allocInfo, commandBuffers.ptr));

    for (commandBuffers) |command_buffer, i| {
        const beginInfo = vk.CommandBufferBeginInfo{
            .sType = vk.StructureType.COMMAND_BUFFER_BEGIN_INFO,
            .flags = vk.CommandBufferUsageFlagBits.SIMULTANEOUS_USE_BIT,
            .pNext = null,
            .pInheritanceInfo = null,
        };

        try checkSuccess(vk.vkBeginCommandBuffer(commandBuffers[i], &beginInfo));

        const clearColor = vk.ClearValue{ .color = vk.ClearColorValue{ .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } };

        const renderPassInfo = vk.RenderPassBeginInfo{
            .sType = vk.StructureType.RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = swapChainFramebuffers[i],
            .renderArea = vk.Rect2D{
                .offset = vk.Offset2D{ .x = 0, .y = 0 },
                .extent = swapChainExtent,
            },
            .clearValueCount = 1,
            .pClearValues = @ptrCast([*]const vk.ClearValue, &clearColor),

            .pNext = null,
        };

        vk.vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, vk.SubpassContents.INLINE);
        {
            vk.vkCmdBindPipeline(commandBuffers[i], vk.PipelineBindPoint.GRAPHICS, graphicsPipeline);

            const vertexBuffers= [1]vk.Buffer{ vertexBuffer };
            const offsets = [1]vk.DeviceSize{ 0 };
            vk.vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffers, &offsets);

            vk.vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);
        }
        vk.vkCmdEndRenderPass(commandBuffers[i]);

        try checkSuccess(vk.vkEndCommandBuffer(commandBuffers[i]));
    }
}

fn createSyncObjects() !void {
    const semaphoreInfo = vk.SemaphoreCreateInfo{
        .sType = vk.StructureType.SEMAPHORE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
    };

    const fenceInfo = vk.FenceCreateInfo{
        .sType = vk.StructureType.FENCE_CREATE_INFO,
        .flags = vk.FenceCreateFlagBits.SIGNALED_BIT,
        .pNext = null,
    };

    var i: usize = 0;
    while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
        try checkSuccess(vk.vkCreateSemaphore(globalDevice, &semaphoreInfo, null, &imageAvailableSemaphores[i]));
        try checkSuccess(vk.vkCreateSemaphore(globalDevice, &semaphoreInfo, null, &renderFinishedSemaphores[i]));
        try checkSuccess(vk.vkCreateFence(globalDevice, &fenceInfo, null, &inFlightFences[i]));
    }
}

fn createCommandPool(allocator: *Allocator) !void {
    const queueFamilyIndices = try findQueueFamilies(allocator, physicalDevice);

    const poolInfo = vk.CommandPoolCreateInfo{
        .sType = vk.StructureType.COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.?,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(vk.vkCreateCommandPool(globalDevice, &poolInfo, null, &commandPool));
}

fn createFramebuffers(allocator: *Allocator) !void {
    swapChainFramebuffers = try allocator.alloc(vk.Framebuffer, swapChainImageViews.len);

    for (swapChainImageViews) |swap_chain_image_view, i| {
        const attachments = [_]vk.ImageView{swap_chain_image_view};

        const framebufferInfo = vk.FramebufferCreateInfo{
            .sType = vk.StructureType.FRAMEBUFFER_CREATE_INFO,
            .renderPass = renderPass,
            .attachmentCount = 1,
            .pAttachments = &attachments,
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .layers = 1,

            .pNext = null,
            .flags = 0,
        };

        try checkSuccess(vk.vkCreateFramebuffer(globalDevice, &framebufferInfo, null, &swapChainFramebuffers[i]));
    }
}

fn createShaderModule(code: []align(@alignOf(u32)) const u8) !vk.ShaderModule {
    const createInfo = vk.ShaderModuleCreateInfo{
        .sType = vk.StructureType.SHADER_MODULE_CREATE_INFO,
        .codeSize = code.len,
        .pCode = std.mem.bytesAsSlice(u32, code).ptr,

        .pNext = null,
        .flags = 0,
    };

    var shaderModule: vk.ShaderModule = undefined;
    try checkSuccess(vk.vkCreateShaderModule(globalDevice, &createInfo, null, &shaderModule));

    return shaderModule;
}

fn createGraphicsPipeline(allocator: *Allocator) !void {
    const vertShaderCode = try std.fs.cwd().readFileAllocAligned(allocator, "shaders\\vert.spv", 1<<20, @alignOf(u32));
    defer allocator.free(vertShaderCode);

    const fragShaderCode = try std.fs.cwd().readFileAllocAligned(allocator, "shaders\\frag.spv", 1<<20, @alignOf(u32));
    defer allocator.free(fragShaderCode);

    const vertShaderModule = try createShaderModule(vertShaderCode);
    defer vk.vkDestroyShaderModule(globalDevice, vertShaderModule, null);

    const fragShaderModule = try createShaderModule(fragShaderCode);
    defer vk.vkDestroyShaderModule(globalDevice, fragShaderModule, null);

    const vertShaderStageInfo = vk.PipelineShaderStageCreateInfo{
        .sType = vk.StructureType.PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = vk.ShaderStageFlagBits.VERTEX_BIT,
        .module = vertShaderModule,
        .pName = "main",
        .pNext = null,
        .flags = 0,
        .pSpecializationInfo = null,
    };

    const fragShaderStageInfo = vk.PipelineShaderStageCreateInfo{
        .sType = vk.StructureType.PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = vk.ShaderStageFlagBits.FRAGMENT_BIT,
        .module = fragShaderModule,
        .pName = "main",
        .pNext = null,
        .flags = 0,
        .pSpecializationInfo = null,
    };

    const shaderStages = [_]vk.PipelineShaderStageCreateInfo{ vertShaderStageInfo, fragShaderStageInfo };

    const bindingDescriptions = Vertex.getBindingDescriptions();
    const attributeDescriptions = Vertex.getAttributeDescriptions();

    const vertexInputInfo = vk.PipelineVertexInputStateCreateInfo{
        .sType = vk.StructureType.PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .vertexAttributeDescriptionCount = attributeDescriptions.len,
        .pVertexBindingDescriptions = &bindingDescriptions,
        .pVertexAttributeDescriptions = &attributeDescriptions,
        .pNext = null,
        .flags = 0,
    };

    const inputAssembly = vk.PipelineInputAssemblyStateCreateInfo{
        .sType = vk.StructureType.PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = vk.PrimitiveTopology.TRIANGLE_LIST,
        .primitiveRestartEnable = vk.FALSE,
        .pNext = null,
        .flags = 0,
    };

    const viewport = [_]vk.Viewport{vk.Viewport{
        .x = 0.0,
        .y = 0.0,
        .width = @intToFloat(f32, swapChainExtent.width),
        .height = @intToFloat(f32, swapChainExtent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    }};

    const scissor = [_]vk.Rect2D{vk.Rect2D{
        .offset = vk.Offset2D{ .x = 0, .y = 0 },
        .extent = swapChainExtent,
    }};

    const viewportState = vk.PipelineViewportStateCreateInfo{
        .sType = vk.StructureType.PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
        .pNext = null,
        .flags = 0,
    };

    const rasterizer = vk.PipelineRasterizationStateCreateInfo{
        .sType = vk.StructureType.PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = vk.FALSE,
        .rasterizerDiscardEnable = vk.FALSE,
        .polygonMode = vk.PolygonMode.FILL,
        .lineWidth = 1.0,
        .cullMode = vk.CullModeFlagBits.BACK_BIT,
        .frontFace = vk.FrontFace.CLOCKWISE,
        .depthBiasEnable = vk.FALSE,
        .pNext = null,
        .flags = 0,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp = 0,
        .depthBiasSlopeFactor = 0,
    };

    const multisampling = vk.PipelineMultisampleStateCreateInfo{
        .sType = vk.StructureType.PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = vk.FALSE,
        .rasterizationSamples = vk.SampleCountFlagBits.T_1_BIT,
        .pNext = null,
        .flags = 0,
        .minSampleShading = 0,
        .pSampleMask = null,
        .alphaToCoverageEnable = 0,
        .alphaToOneEnable = 0,
    };

    const colorBlendAttachment = vk.PipelineColorBlendAttachmentState{
        .colorWriteMask = vk.ColorComponentFlagBits.R_BIT | vk.ColorComponentFlagBits.G_BIT | vk.ColorComponentFlagBits.B_BIT | vk.ColorComponentFlagBits.A_BIT,
        .blendEnable = vk.FALSE,

        .srcColorBlendFactor = vk.BlendFactor.ZERO,
        .dstColorBlendFactor = vk.BlendFactor.ZERO,
        .colorBlendOp = vk.BlendOp.ADD,
        .srcAlphaBlendFactor = vk.BlendFactor.ZERO,
        .dstAlphaBlendFactor = vk.BlendFactor.ZERO,
        .alphaBlendOp = vk.BlendOp.ADD,
    };

    const colorBlending = vk.PipelineColorBlendStateCreateInfo{
        .sType = vk.StructureType.PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = vk.FALSE,
        .logicOp = vk.LogicOp.COPY,
        .attachmentCount = 1,
        .pAttachments = @ptrCast([*]const vk.PipelineColorBlendAttachmentState, &colorBlendAttachment),
        .blendConstants = [_]f32{ 0, 0, 0, 0 },
        .pNext = null,
        .flags = 0,
    };

    const pipelineLayoutInfo = vk.PipelineLayoutCreateInfo{
        .sType = vk.StructureType.PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0,
        .pNext = null,
        .flags = 0,
        .pSetLayouts = undefined,
        .pPushConstantRanges = undefined,
    };

    try checkSuccess(vk.vkCreatePipelineLayout(globalDevice, &pipelineLayoutInfo, null, &pipelineLayout));

    const pipelineInfo = [_]vk.GraphicsPipelineCreateInfo{vk.GraphicsPipelineCreateInfo{
        .sType = vk.StructureType.GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = @intCast(u32, shaderStages.len),
        .pStages = &shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .layout = pipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
        .basePipelineHandle = null,
        .pNext = null,
        .flags = 0,
        .pTessellationState = null,
        .pDepthStencilState = null,
        .pDynamicState = null,
        .basePipelineIndex = 0,
    }};

    try checkSuccess(vk.vkCreateGraphicsPipelines(
        globalDevice,
        null,
        @intCast(u32, pipelineInfo.len),
        &pipelineInfo,
        null,
        @as(*[1]vk.Pipeline, &graphicsPipeline),
    ));
}

fn createRenderPass() !void {
    const colorAttachment = vk.AttachmentDescription{
        .format = swapChainImageFormat,
        .samples = vk.SampleCountFlagBits.T_1_BIT,
        .loadOp = vk.AttachmentLoadOp.DONT_CARE, // Was LOAD
        .storeOp = vk.AttachmentStoreOp.STORE,
        .stencilLoadOp = vk.AttachmentLoadOp.DONT_CARE,
        .stencilStoreOp = vk.AttachmentStoreOp.DONT_CARE,
        .initialLayout = vk.ImageLayout.UNDEFINED,
        .finalLayout = vk.ImageLayout.PRESENT_SRC_KHR,
        .flags = 0,
    };

    const colorAttachmentRef = vk.AttachmentReference{
        .attachment = 0,
        .layout = vk.ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
    };

    const subpass = [_]vk.SubpassDescription{vk.SubpassDescription{
        .pipelineBindPoint = vk.PipelineBindPoint.GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = @ptrCast([*]const vk.AttachmentReference, &colorAttachmentRef),

        .flags = 0,
        .inputAttachmentCount = 0,
        .pInputAttachments = undefined,
        .pResolveAttachments = null,
        .pDepthStencilAttachment = null,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = undefined,
    }};

    const dependency = [_]vk.SubpassDependency{vk.SubpassDependency{
        .srcSubpass = vk.SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = vk.AccessFlagBits.COLOR_ATTACHMENT_READ_BIT | vk.AccessFlagBits.COLOR_ATTACHMENT_WRITE_BIT,

        .dependencyFlags = 0,
    }};

    const renderPassInfo = vk.RenderPassCreateInfo{
        .sType = vk.StructureType.RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = @ptrCast(*const [1]vk.AttachmentDescription, &colorAttachment),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(vk.vkCreateRenderPass(globalDevice, &renderPassInfo, null, &renderPass));
}

fn createSwapChain(allocator: *Allocator) !void {
    var swapChainSupport = try querySwapChainSupport(allocator, physicalDevice);
    defer swapChainSupport.deinit();

    const surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats.items);
    const presentMode = chooseSwapPresentMode(swapChainSupport.presentModes.items);
    const extent = chooseSwapExtent(swapChainSupport.capabilities);

    var imageCount: u32 = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 and
        imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    const indices = try findQueueFamilies(allocator, physicalDevice);
    const queueFamilyIndices = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };

    const different_families = indices.graphicsFamily.? != indices.presentFamily.?;

    var createInfo = vk.SwapchainCreateInfoKHR{
        .sType = vk.StructureType.SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,

        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk.ImageUsageFlagBits.COLOR_ATTACHMENT_BIT,

        .imageSharingMode = if (different_families) vk.SharingMode.CONCURRENT else vk.SharingMode.EXCLUSIVE,
        .queueFamilyIndexCount = if (different_families) @as(u32, 2) else @as(u32, 0),
        .pQueueFamilyIndices = if (different_families) &queueFamilyIndices else &([_]u32{ 0, 0 }),

        .preTransform = swapChainSupport.capabilities.currentTransform,
        .compositeAlpha = vk.CompositeAlphaFlagBitsKHR.OPAQUE_BIT,
        .presentMode = presentMode,
        .clipped = vk.TRUE,

        .oldSwapchain = null,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(vk.vkCreateSwapchainKHR(globalDevice, &createInfo, null, &swapChain));

    try checkSuccess(vk.vkGetSwapchainImagesKHR(globalDevice, swapChain, &imageCount, null));
    swapChainImages = try allocator.alloc(vk.Image, imageCount);
    try checkSuccess(vk.vkGetSwapchainImagesKHR(globalDevice, swapChain, &imageCount, swapChainImages.ptr));

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

fn chooseSwapSurfaceFormat(availableFormats: []vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    if (availableFormats.len == 1 and availableFormats[0].format == vk.Format.UNDEFINED) {
        return vk.SurfaceFormatKHR{
            .format = vk.Format.B8G8R8A8_UNORM,
            .colorSpace = vk.ColorSpaceKHR.SRGB_NONLINEAR,
        };
    }

    for (availableFormats) |availableFormat| {
        if (availableFormat.format == vk.Format.B8G8R8A8_UNORM and
            availableFormat.colorSpace == vk.ColorSpaceKHR.SRGB_NONLINEAR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

fn querySwapChainSupport(allocator: *Allocator, device: vk.PhysicalDevice) !SwapChainSupportDetails {
    var details = SwapChainSupportDetails.init(allocator);

    try checkSuccess(vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities));

    var formatCount: u32 = undefined;
    try checkSuccess(vk.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, null));

    if (formatCount != 0) {
        try details.formats.resize(formatCount);
        try checkSuccess(vk.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.items.ptr));
    }

    var presentModeCount: u32 = undefined;
    try checkSuccess(vk.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, null));

    if (presentModeCount != 0) {
        try details.presentModes.resize(presentModeCount);
        try checkSuccess(vk.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.items.ptr));
    }

    return details;
}

fn chooseSwapPresentMode(availablePresentModes: []vk.PresentModeKHR) vk.PresentModeKHR {
    var bestMode: vk.PresentModeKHR = vk.PresentModeKHR.FIFO;

    for (availablePresentModes) |availablePresentMode| {
        if (availablePresentMode == vk.PresentModeKHR.MAILBOX) {
            return availablePresentMode;
        } else if (availablePresentMode == vk.PresentModeKHR.IMMEDIATE) {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

fn chooseSwapExtent(capabilities: vk.SurfaceCapabilitiesKHR) vk.Extent2D {
    if (capabilities.currentExtent.width != maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        var actualExtent = vk.Extent2D{
            .width = WIDTH,
            .height = HEIGHT,
        };

        actualExtent.width = std.math.max(capabilities.minImageExtent.width, std.math.min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std.math.max(capabilities.minImageExtent.height, std.math.min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

fn createImageViews(allocator: *Allocator) !void {
    swapChainImageViews = try allocator.alloc(vk.ImageView, swapChainImages.len);
    errdefer allocator.free(swapChainImageViews);

    for (swapChainImages) |swap_chain_image, i| {
        const createInfo = vk.ImageViewCreateInfo{
            .sType = vk.StructureType.IMAGE_VIEW_CREATE_INFO,
            .image = swap_chain_image,
            .viewType = vk.ImageViewType.T_2D,
            .format = swapChainImageFormat,
            .components = vk.ComponentMapping{
                .r = vk.ComponentSwizzle.IDENTITY,
                .g = vk.ComponentSwizzle.IDENTITY,
                .b = vk.ComponentSwizzle.IDENTITY,
                .a = vk.ComponentSwizzle.IDENTITY,
            },
            .subresourceRange = vk.ImageSubresourceRange{
                .aspectMask = vk.ImageAspectFlagBits.COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },

            .pNext = null,
            .flags = 0,
        };

        try checkSuccess(vk.vkCreateImageView(globalDevice, &createInfo, null, &swapChainImageViews[i]));
    }
}

fn createLogicalDevice(allocator: *Allocator) !void {
    const indices = try findQueueFamilies(allocator, physicalDevice);

    var queueCreateInfos = std.ArrayList(vk.DeviceQueueCreateInfo).init(allocator);
    defer queueCreateInfos.deinit();
    const allQueueFamilies = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };
    const uniqueQueueFamilies = if (indices.graphicsFamily.? == indices.presentFamily.?)
        allQueueFamilies[0..1]
    else
        allQueueFamilies[0..2];

    var queuePriority: f32 = 1.0;
    for (uniqueQueueFamilies) |queueFamily| {
        const queueCreateInfo = vk.DeviceQueueCreateInfo{
            .sType = vk.StructureType.DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = @ptrCast([*]const f32, &queuePriority),
            .pNext = null,
            .flags = 0,
        };
        try queueCreateInfos.append(queueCreateInfo);
    }

    const deviceFeatures = vk.PhysicalDeviceFeatures{
        .robustBufferAccess = 0,
        .fullDrawIndexUint32 = 0,
        .imageCubeArray = 0,
        .independentBlend = 0,
        .geometryShader = 0,
        .tessellationShader = 0,
        .sampleRateShading = 0,
        .dualSrcBlend = 0,
        .logicOp = 0,
        .multiDrawIndirect = 0,
        .drawIndirectFirstInstance = 0,
        .depthClamp = 0,
        .depthBiasClamp = 0,
        .fillModeNonSolid = 0,
        .depthBounds = 0,
        .wideLines = 0,
        .largePoints = 0,
        .alphaToOne = 0,
        .multiViewport = 0,
        .samplerAnisotropy = 0,
        .textureCompressionETC2 = 0,
        .textureCompressionASTC_LDR = 0,
        .textureCompressionBC = 0,
        .occlusionQueryPrecise = 0,
        .pipelineStatisticsQuery = 0,
        .vertexPipelineStoresAndAtomics = 0,
        .fragmentStoresAndAtomics = 0,
        .shaderTessellationAndGeometryPointSize = 0,
        .shaderImageGatherExtended = 0,
        .shaderStorageImageExtendedFormats = 0,
        .shaderStorageImageMultisample = 0,
        .shaderStorageImageReadWithoutFormat = 0,
        .shaderStorageImageWriteWithoutFormat = 0,
        .shaderUniformBufferArrayDynamicIndexing = 0,
        .shaderSampledImageArrayDynamicIndexing = 0,
        .shaderStorageBufferArrayDynamicIndexing = 0,
        .shaderStorageImageArrayDynamicIndexing = 0,
        .shaderClipDistance = 0,
        .shaderCullDistance = 0,
        .shaderFloat64 = 0,
        .shaderInt64 = 0,
        .shaderInt16 = 0,
        .shaderResourceResidency = 0,
        .shaderResourceMinLod = 0,
        .sparseBinding = 0,
        .sparseResidencyBuffer = 0,
        .sparseResidencyImage2D = 0,
        .sparseResidencyImage3D = 0,
        .sparseResidency2Samples = 0,
        .sparseResidency4Samples = 0,
        .sparseResidency8Samples = 0,
        .sparseResidency16Samples = 0,
        .sparseResidencyAliased = 0,
        .variableMultisampleRate = 0,
        .inheritedQueries = 0,
    };

    const createInfo = vk.DeviceCreateInfo{
        .sType = vk.StructureType.DEVICE_CREATE_INFO,

        .queueCreateInfoCount = @intCast(u32, queueCreateInfos.items.len),
        .pQueueCreateInfos = queueCreateInfos.items.ptr,

        .pEnabledFeatures = &deviceFeatures,

        .enabledExtensionCount = @intCast(u32, deviceExtensions.len),
        .ppEnabledExtensionNames = &deviceExtensions,
        .enabledLayerCount = if (enableValidationLayers) @intCast(u32, validationLayers.len) else 0,
        .ppEnabledLayerNames = if (enableValidationLayers) &validationLayers else null,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(vk.vkCreateDevice(physicalDevice, &createInfo, null, &globalDevice));

    vk.vkGetDeviceQueue(globalDevice, indices.graphicsFamily.?, 0, &graphicsQueue);
    vk.vkGetDeviceQueue(globalDevice, indices.presentFamily.?, 0, &presentQueue);
}

fn debugCallback(
    flags: vk.DebugReportFlagsEXT,
    objType: vk.DebugReportObjectTypeEXT,
    obj: u64,
    location: usize,
    code: i32,
    layerPrefix: ?[*]const u8,
    msg: ?[*]const u8,
    userData: ?*c_void,
) callconv(.C) vk.Bool32 {
    std.debug.warn("validation layer: {s}\n", .{@ptrCast([*:0] const u8, msg)});
    return vk.FALSE;
}

fn setupDebugCallback() error{FailedToSetUpDebugCallback}!void {
    if (!enableValidationLayers) return;

    var createInfo = vk.DebugReportCallbackCreateInfoEXT{
        .sType = vk.StructureType.DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
        .flags = vk.DebugReportFlagBitsEXT.ERROR_BIT | vk.DebugReportFlagBitsEXT.WARNING_BIT,
        .pfnCallback = debugCallback,
        .pNext = null,
        .pUserData = null,
    };

    if (CreateDebugReportCallbackEXT(&createInfo, null, &callback) != vk.Result.SUCCESS) {
        return error.FailedToSetUpDebugCallback;
    }
}

fn DestroyDebugReportCallbackEXT(
    pAllocator: ?*const vk.AllocationCallbacks,
) void {
    const func = @ptrCast(?@TypeOf(vk.vkDestroyDebugReportCallbackEXT), vk.GetInstanceProcAddr(
        instance,
        "vkDestroyDebugReportCallbackEXT",
    )) orelse unreachable;
    func(instance, callback, pAllocator);
}

fn CreateDebugReportCallbackEXT(
    pCreateInfo: *const vk.DebugReportCallbackCreateInfoEXT,
    pAllocator: ?*const vk.AllocationCallbacks,
    pCallback: *vk.DebugReportCallbackEXT,
) vk.Result {
    const func = @ptrCast(?@TypeOf(vk.vkCreateDebugReportCallbackEXT), vk.GetInstanceProcAddr(
        instance,
        "vkCreateDebugReportCallbackEXT",
    )) orelse return .ERROR_EXTENSION_NOT_PRESENT;
    return func(instance, pCreateInfo, pAllocator, pCallback);
}

fn checkSuccess(result: vk.Result) !void {
    switch (result) {
        vk.Result.SUCCESS => {},
        else => return error.Unexpected,
    }
}

fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layerCount: u32 = undefined;

    try checkSuccess(vk.vkEnumerateInstanceLayerProperties(&layerCount, null));

    const availableLayers = try allocator.alloc(vk.LayerProperties, layerCount);
    defer allocator.free(availableLayers);

    try checkSuccess(vk.vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.ptr));

    for (validationLayers) |layerName| {
        var layerFound = false;

        for (availableLayers) |layerProperties| {
            if (std.cstr.cmp(layerName, @ptrCast([*:0]const u8, &layerProperties.layerName)) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

fn drawFrame() !void {
    try checkSuccess(vk.vkWaitForFences(globalDevice, 1, @as(*[1]vk.Fence, &inFlightFences[currentFrame]), vk.TRUE, maxInt(u64)));
    try checkSuccess(vk.vkResetFences(globalDevice, 1, @as(*[1]vk.Fence, &inFlightFences[currentFrame])));

    var imageIndex: u32 = undefined;
    try checkSuccess(vk.vkAcquireNextImageKHR(globalDevice, swapChain, maxInt(u64), imageAvailableSemaphores[currentFrame], null, &imageIndex));

    var waitSemaphores = [_]vk.Semaphore{imageAvailableSemaphores[currentFrame]};
    var waitStages = [_]vk.PipelineStageFlags{vk.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT_BIT};

    const signalSemaphores = [_]vk.Semaphore{renderFinishedSemaphores[currentFrame]};

    var submitInfo = [_]vk.SubmitInfo{vk.SubmitInfo{
        .sType = vk.StructureType.SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &waitSemaphores,
        .pWaitDstStageMask = &waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = commandBuffers.ptr + imageIndex,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signalSemaphores,

        .pNext = null,
    }};

    try checkSuccess(vk.vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));

    const swapChains = [_]vk.SwapchainKHR{swapChain};
    const presentInfo = vk.PresentInfoKHR{
        .sType = vk.StructureType.PRESENT_INFO_KHR,

        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signalSemaphores,

        .swapchainCount = 1,
        .pSwapchains = &swapChains,

        .pImageIndices = @ptrCast(*[1]u32, &imageIndex),

        .pNext = null,
        .pResults = null,
    };

    try checkSuccess(vk.vkQueuePresentKHR(presentQueue, &presentInfo));

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

fn hash_cstr(a: [*]const u8) u32 {
    // FNV 32-bit hash
    var h: u32 = 2166136261;
    var i: usize = 0;
    while (a[i] != 0) : (i += 1) {
        h ^= a[i];
        h *%= 16777619;
    }
    return h;
}

fn eql_cstr(a: [*]const u8, b: [*]const u8) bool {
    return std.cstr.cmp(@ptrCast([*:0] const u8, a), @ptrCast([*:0] const u8, b) ) == 0;
}