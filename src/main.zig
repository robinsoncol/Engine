const std = @import("std");
const assert = std.debug.assert;
const mem = std.mem;
const Allocator = mem.Allocator;
const maxInt = std.math.maxInt;

const v = @import("vulkan_core.zig");
const g = @import("glfw.zig");

const WIDTH = 1280;
const HEIGHT = 720;

const MAX_FRAMES_IN_FLIGHT = 2;

const enableValidationLayers = std.debug.runtime_safety;
const validationLayers = [_][*:0]const u8{"VK_LAYER_LUNARG_standard_validation"};
const deviceExtensions = [_][*:0]const u8{v.KHR_SWAPCHAIN_EXTENSION_NAME};

var currentFrame: usize = 0;
var instance: v.Instance = undefined;
var callback: v.DebugReportCallbackEXT = undefined;
var surface: v.SurfaceKHR = undefined;
var physicalDevice: v.PhysicalDevice = undefined;
var global_device: v.Device = undefined;
var graphicsQueue: v.Queue = undefined;
var presentQueue: v.Queue = undefined;
var swapChainImages: []v.Image = undefined;
var swapChain: v.SwapchainKHR = undefined;
var swapChainImageFormat: v.Format = undefined;
var swapChainExtent: v.Extent2D = undefined;
var swapChainImageViews: []v.ImageView = undefined;
var renderPass: v.RenderPass = undefined;
var pipelineLayout: v.PipelineLayout = undefined;
var graphicsPipeline: v.Pipeline = undefined;
var swapChainFramebuffers: []v.Framebuffer = undefined;
var commandPool: v.CommandPool = undefined;
var commandBuffers: []v.CommandBuffer = undefined;

var imageAvailableSemaphores: [MAX_FRAMES_IN_FLIGHT]v.Semaphore = undefined;
var renderFinishedSemaphores: [MAX_FRAMES_IN_FLIGHT]v.Semaphore = undefined;
var inFlightFences: [MAX_FRAMES_IN_FLIGHT]v.Fence = undefined;

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
    capabilities: v.SurfaceCapabilitiesKHR,
    formats: std.ArrayList(v.SurfaceFormatKHR),
    presentModes: std.ArrayList(v.PresentModeKHR),

    fn init(allocator: *Allocator) SwapChainSupportDetails {
        var result = SwapChainSupportDetails{
            .capabilities = undefined,
            .formats = std.ArrayList(v.SurfaceFormatKHR).init(allocator),
            .presentModes = std.ArrayList(v.PresentModeKHR).init(allocator),
        };
        const slice = @sliceToBytes(@as(*[1]v.SurfaceCapabilitiesKHR, &result.capabilities)[0..1]);
        std.mem.set(u8, slice, 0);
        return result;
    }

    fn deinit(self: *SwapChainSupportDetails) void {
        self.formats.deinit();
        self.presentModes.deinit();
    }
};

pub fn main() !void {
    if (g.glfwInit() == 0) return error.GlfwInitFailed;
    defer g.glfwTerminate();

    g.glfwWindowHint(g.GLFW_CLIENT_API, g.GLFW_NO_API);
    g.glfwWindowHint(g.GLFW_RESIZABLE, g.GLFW_FALSE);

    const window = g.glfwCreateWindow(WIDTH, HEIGHT, "Zig Vulkan Triangle", null, null) orelse return error.GlfwCreateWindowFailed;
    defer g.glfwDestroyWindow(window);

    const allocator = std.heap.c_allocator;
    try initVulkan(allocator, window);

    while (g.glfwWindowShouldClose(window) == 0) {
        g.glfwPollEvents();
        try drawFrame();
    }
    try checkSuccess(v.vkDeviceWaitIdle(global_device));

    cleanup();
}

fn cleanup() void {
    var i: usize = 0;
    while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
        v.vkDestroySemaphore(global_device, renderFinishedSemaphores[i], null);
        v.vkDestroySemaphore(global_device, imageAvailableSemaphores[i], null);
        v.vkDestroyFence(global_device, inFlightFences[i], null);
    }

    v.vkDestroyCommandPool(global_device, commandPool, null);

    for (swapChainFramebuffers) |framebuffer| {
        v.vkDestroyFramebuffer(global_device, framebuffer, null);
    }

    v.vkDestroyPipeline(global_device, graphicsPipeline, null);
    v.vkDestroyPipelineLayout(global_device, pipelineLayout, null);
    v.vkDestroyRenderPass(global_device, renderPass, null);

    for (swapChainImageViews) |imageView| {
        v.vkDestroyImageView(global_device, imageView, null);
    }

    v.vkDestroySwapchainKHR(global_device, swapChain, null);
    v.vkDestroyDevice(global_device, null);

    if (enableValidationLayers) {
        // DestroyDebugReportCallbackEXT(null);
    }

    v.vkDestroySurfaceKHR(instance, surface, null);
    v.vkDestroyInstance(instance, null);
}

fn initVulkan(allocator: *Allocator, window: *g.GLFWwindow) !void {
    // Helper function to list available extensions
    // try listExtensions(allocator);

    try createInstance(allocator);

    // TODO: The debug callback function doesn't seem to work. Get back to this!
    try setupDebugCallback();

    try createSurface(window);
    try pickPhysicalDevice(allocator);
    try createLogicalDevice(allocator);
    try createSwapChain(allocator);
    try createImageViews(allocator);
    try createRenderPass();
    try createGraphicsPipeline(allocator);
    try createFramebuffers(allocator);
    try createCommandPool(allocator);
    try createCommandBuffers(allocator);
    try createSyncObjects();
}

fn listExtensions(allocator: *Allocator) !void {
    var extensionCount : u32 = undefined;

    try checkSuccess(v.vkEnumerateInstanceExtensionProperties(null, &extensionCount, null));

    const extensions = try allocator.alloc(v.ExtensionProperties, extensionCount);
    defer allocator.free(extensions);

    try checkSuccess(v.vkEnumerateInstanceExtensionProperties(null, &extensionCount, extensions.ptr));

    for (extensions) |extension, i| {
        std.debug.warn("Extensions({}): {}\n", .{ i, extension.extensionName });
    }
}

fn createInstance(allocator: *Allocator) !void {
    if (enableValidationLayers) {
        if (!(try checkValidationLayerSupport(allocator))) {
            return error.ValidationLayerRequestedButNotAvailable;
        }
    }

    const appInfo = v.ApplicationInfo{
        .sType = v.StructureType.APPLICATION_INFO,
        // We tell the instance which version of vulkan we support
        .apiVersion = v.MAKE_VERSION(1, 2, 0),
        .applicationVersion = v.MAKE_VERSION(1, 0, 0),
        .engineVersion = v.MAKE_VERSION(1, 0, 0),
        .pApplicationName = null,
        .pEngineName = null,
        .pNext = null,
    };

    const extensions = try getRequiredExtensions(allocator);
    defer allocator.free(extensions);

    const createInfo = v.InstanceCreateInfo{
        .sType = v.StructureType.INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,

        .enabledExtensionCount = @intCast(u32, extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,

        // Here we pass the layers that we want to enable
        .enabledLayerCount = if (enableValidationLayers) @intCast(u32, validationLayers.len) else 0,
        .ppEnabledLayerNames = if (enableValidationLayers) &validationLayers else null,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(v.vkCreateInstance(&createInfo, null, &instance));
}

/// caller must free returned memory
fn getRequiredExtensions(allocator: *Allocator) ![][*]const u8 {
    var glfwExtensionCount: u32 = 0;
    var glfwExtensions: [*]const [*]const u8 = g.glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    var extensions = std.ArrayList([*]const u8).init(allocator);
    errdefer extensions.deinit();

    try extensions.appendSlice(glfwExtensions[0..glfwExtensionCount]);

    if (enableValidationLayers) {
        // TODO: Got this from the vulkan tutorial. Find out if we need this and/or
        // if this is better/worse than VK_EXT_DEBUG_REPORT_EXTENSION_NAME.
        // try extensions.append(v.VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        try extensions.append(v.EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions.toOwnedSlice();
}

fn createSurface(window: *g.GLFWwindow) !void {
    if (g.glfwCreateWindowSurface(instance, window, null, &surface) != v.Result.SUCCESS) {
        return error.FailedToCreateWindowSurface;
    }
}

// Currently reviewing...

fn pickPhysicalDevice(allocator: *Allocator) !void {
    var deviceCount: u32 = 0;

    // Note: The instance has the required api version, which might limit the available
    // device(s)
    try checkSuccess(v.vkEnumeratePhysicalDevices(instance, &deviceCount, null));

    if (deviceCount == 0) {
        return error.FailedToFindGPUsWithVulkanSupport;
    }

    const devices = try allocator.alloc(v.PhysicalDevice, deviceCount);
    defer allocator.free(devices);
    try checkSuccess(v.vkEnumeratePhysicalDevices(instance, &deviceCount, devices.ptr));

    physicalDevice = for (devices) |device| {
        if (try isDeviceSuitable(allocator, device)) {
            break device;
        }
    } else return error.FailedToFindSuitableGPU;
}

fn isDeviceSuitable(allocator: *Allocator, device: v.PhysicalDevice) !bool {
    const indices = try findQueueFamilies(allocator, device);

    const extensionsSupported = try checkDeviceExtensionSupport(allocator, device);

    var swapChainAdequate = false;
    if (extensionsSupported) {
        var swapChainSupport = try querySwapChainSupport(allocator, device);
        defer swapChainSupport.deinit();
        swapChainAdequate = swapChainSupport.formats.len != 0 and swapChainSupport.presentModes.len != 0;
    }

    return indices.isComplete() and extensionsSupported and swapChainAdequate;
}

// All operations are processed in a queue. We return the first queue family that meets
// our needs.
fn findQueueFamilies(allocator: *Allocator, device: v.PhysicalDevice) !QueueFamilyIndices {
    var indices = QueueFamilyIndices.init();

    var queueFamilyCount: u32 = 0;
    v.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);

    const queueFamilies = try allocator.alloc(v.QueueFamilyProperties, queueFamilyCount);
    defer allocator.free(queueFamilies);
    v.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.ptr);

    var i: u32 = 0;
    for (queueFamilies) |queueFamily| {
        if (queueFamily.queueCount > 0 and
            queueFamily.queueFlags & v.QueueFlagBits.GRAPHICS_BIT != 0)
        {
            indices.graphicsFamily = i;
        }

        var presentSupport: v.Bool32 = 0;
        try checkSuccess(v.vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport));

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

fn checkDeviceExtensionSupport(allocator: *Allocator, device: v.PhysicalDevice) !bool {
    var extensionCount: u32 = undefined;
    try checkSuccess(v.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, null));

    const availableExtensions = try allocator.alloc(v.ExtensionProperties, extensionCount);
    defer allocator.free(availableExtensions);
    try checkSuccess(v.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, availableExtensions.ptr));

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

// To Review...

fn createCommandBuffers(allocator: *Allocator) !void {
    commandBuffers = try allocator.alloc(v.CommandBuffer, swapChainFramebuffers.len);

    const allocInfo = v.CommandBufferAllocateInfo{
        .sType = v.StructureType.COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = v.CommandBufferLevel.PRIMARY,
        .commandBufferCount = @intCast(u32, commandBuffers.len),
        .pNext = null,
    };

    try checkSuccess(v.vkAllocateCommandBuffers(global_device, &allocInfo, commandBuffers.ptr));

    for (commandBuffers) |command_buffer, i| {
        const beginInfo = v.CommandBufferBeginInfo{
            .sType = v.StructureType.COMMAND_BUFFER_BEGIN_INFO,
            .flags = v.CommandBufferUsageFlagBits.SIMULTANEOUS_USE_BIT,
            .pNext = null,
            .pInheritanceInfo = null,
        };

        try checkSuccess(v.vkBeginCommandBuffer(commandBuffers[i], &beginInfo));

        const clearColor = v.ClearValue{ .color = v.ClearColorValue{ .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } };

        const renderPassInfo = v.RenderPassBeginInfo{
            .sType = v.StructureType.RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = swapChainFramebuffers[i],
            .renderArea = v.Rect2D{
                .offset = v.Offset2D{ .x = 0, .y = 0 },
                .extent = swapChainExtent,
            },
            .clearValueCount = 1,
            .pClearValues = @ptrCast([*]const v.ClearValue, &clearColor),

            .pNext = null,
        };

        v.vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, v.SubpassContents.INLINE);
        {
            v.vkCmdBindPipeline(commandBuffers[i], v.PipelineBindPoint.GRAPHICS, graphicsPipeline);
            v.vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);
        }
        v.vkCmdEndRenderPass(commandBuffers[i]);

        try checkSuccess(v.vkEndCommandBuffer(commandBuffers[i]));
    }
}

fn createSyncObjects() !void {
    const semaphoreInfo = v.SemaphoreCreateInfo{
        .sType = v.StructureType.SEMAPHORE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
    };

    const fenceInfo = v.FenceCreateInfo{
        .sType = v.StructureType.FENCE_CREATE_INFO,
        .flags = v.FenceCreateFlagBits.SIGNALED_BIT,
        .pNext = null,
    };

    var i: usize = 0;
    while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
        try checkSuccess(v.vkCreateSemaphore(global_device, &semaphoreInfo, null, &imageAvailableSemaphores[i]));
        try checkSuccess(v.vkCreateSemaphore(global_device, &semaphoreInfo, null, &renderFinishedSemaphores[i]));
        try checkSuccess(v.vkCreateFence(global_device, &fenceInfo, null, &inFlightFences[i]));
    }
}

fn createCommandPool(allocator: *Allocator) !void {
    const queueFamilyIndices = try findQueueFamilies(allocator, physicalDevice);

    const poolInfo = v.CommandPoolCreateInfo{
        .sType = v.StructureType.COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.?,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(v.vkCreateCommandPool(global_device, &poolInfo, null, &commandPool));
}

fn createFramebuffers(allocator: *Allocator) !void {
    swapChainFramebuffers = try allocator.alloc(v.Framebuffer, swapChainImageViews.len);

    for (swapChainImageViews) |swap_chain_image_view, i| {
        const attachments = [_]v.ImageView{swap_chain_image_view};

        const framebufferInfo = v.FramebufferCreateInfo{
            .sType = v.StructureType.FRAMEBUFFER_CREATE_INFO,
            .renderPass = renderPass,
            .attachmentCount = 1,
            .pAttachments = &attachments,
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .layers = 1,

            .pNext = null,
            .flags = 0,
        };

        try checkSuccess(v.vkCreateFramebuffer(global_device, &framebufferInfo, null, &swapChainFramebuffers[i]));
    }
}

fn createShaderModule(code: []align(@alignOf(u32)) const u8) !v.ShaderModule {
    const createInfo = v.ShaderModuleCreateInfo{
        .sType = v.StructureType.SHADER_MODULE_CREATE_INFO,
        .codeSize = code.len,
        .pCode = @bytesToSlice(u32, code).ptr,

        .pNext = null,
        .flags = 0,
    };

    var shaderModule: v.ShaderModule = undefined;
    try checkSuccess(v.vkCreateShaderModule(global_device, &createInfo, null, &shaderModule));

    return shaderModule;
}

fn createGraphicsPipeline(allocator: *Allocator) !void {
    const vertShaderCode = try std.fs.cwd().readFileAllocAligned(allocator, "shaders\\vert.spv", 1<<20, @alignOf(u32));
    defer allocator.free(vertShaderCode);

    const fragShaderCode = try std.fs.cwd().readFileAllocAligned(allocator, "shaders\\frag.spv", 1<<20, @alignOf(u32));
    defer allocator.free(fragShaderCode);

    const vertShaderModule = try createShaderModule(vertShaderCode);
    const fragShaderModule = try createShaderModule(fragShaderCode);

    const vertShaderStageInfo = v.PipelineShaderStageCreateInfo{
        .sType = v.StructureType.PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = v.ShaderStageFlagBits.VERTEX_BIT,
        .module = vertShaderModule,
        .pName = "main",

        .pNext = null,
        .flags = 0,
        .pSpecializationInfo = null,
    };

    const fragShaderStageInfo = v.PipelineShaderStageCreateInfo{
        .sType = v.StructureType.PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = v.ShaderStageFlagBits.FRAGMENT_BIT,
        .module = fragShaderModule,
        .pName = "main",
        .pNext = null,
        .flags = 0,

        .pSpecializationInfo = null,
    };

    const shaderStages = [_]v.PipelineShaderStageCreateInfo{ vertShaderStageInfo, fragShaderStageInfo };

    const vertexInputInfo = v.PipelineVertexInputStateCreateInfo{
        .sType = v.StructureType.PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0,

        // .pVertexBindingDescriptions = null,
        // .pVertexAttributeDescriptions = null,
        .pNext = null,
        .flags = 0,
    };

    const inputAssembly = v.PipelineInputAssemblyStateCreateInfo{
        .sType = v.StructureType.PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = v.PrimitiveTopology.TRIANGLE_LIST,
        .primitiveRestartEnable = v.FALSE,
        .pNext = null,
        .flags = 0,
    };

    const viewport = [_]v.Viewport{v.Viewport{
        .x = 0.0,
        .y = 0.0,
        .width = @intToFloat(f32, swapChainExtent.width),
        .height = @intToFloat(f32, swapChainExtent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    }};

    const scissor = [_]v.Rect2D{v.Rect2D{
        .offset = v.Offset2D{ .x = 0, .y = 0 },
        .extent = swapChainExtent,
    }};

    const viewportState = v.PipelineViewportStateCreateInfo{
        .sType = v.StructureType.PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,

        .pNext = null,
        .flags = 0,
    };

    const rasterizer = v.PipelineRasterizationStateCreateInfo{
        .sType = v.StructureType.PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = v.FALSE,
        .rasterizerDiscardEnable = v.FALSE,
        .polygonMode = v.PolygonMode.FILL,
        .lineWidth = 1.0,
        .cullMode = v.CullModeFlagBits.BACK_BIT,
        .frontFace = v.FrontFace.CLOCKWISE,
        .depthBiasEnable = v.FALSE,

        .pNext = null,
        .flags = 0,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp = 0,
        .depthBiasSlopeFactor = 0,
    };

    const multisampling = v.PipelineMultisampleStateCreateInfo{
        .sType = v.StructureType.PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = v.FALSE,
        .rasterizationSamples = v.SampleCountFlagBits.T_1_BIT,
        .pNext = null,
        .flags = 0,
        .minSampleShading = 0,
        .pSampleMask = null,
        .alphaToCoverageEnable = 0,
        .alphaToOneEnable = 0,
    };

    const colorBlendAttachment = v.PipelineColorBlendAttachmentState{
        .colorWriteMask = v.ColorComponentFlagBits.R_BIT | v.ColorComponentFlagBits.G_BIT | v.ColorComponentFlagBits.B_BIT | v.ColorComponentFlagBits.A_BIT,
        .blendEnable = v.FALSE,

        .srcColorBlendFactor = v.BlendFactor.ZERO,
        .dstColorBlendFactor = v.BlendFactor.ZERO,
        .colorBlendOp = v.BlendOp.ADD,
        .srcAlphaBlendFactor = v.BlendFactor.ZERO,
        .dstAlphaBlendFactor = v.BlendFactor.ZERO,
        .alphaBlendOp = v.BlendOp.ADD,
    };

    const colorBlending = v.PipelineColorBlendStateCreateInfo{
        .sType = v.StructureType.PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = v.FALSE,
        .logicOp = v.LogicOp.COPY,
        .attachmentCount = 1,
        .pAttachments = @ptrCast([*]const v.PipelineColorBlendAttachmentState, &colorBlendAttachment),
        .blendConstants = [_]f32{ 0, 0, 0, 0 },

        .pNext = null,
        .flags = 0,
    };

    const pipelineLayoutInfo = v.PipelineLayoutCreateInfo{
        .sType = v.StructureType.PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0,
        .pNext = null,
        .flags = 0,
        // .pSetLayouts = null,
        // .pPushConstantRanges = null,
    };

    try checkSuccess(v.vkCreatePipelineLayout(global_device, &pipelineLayoutInfo, null, &pipelineLayout));

    const pipelineInfo = [_]v.GraphicsPipelineCreateInfo{v.GraphicsPipelineCreateInfo{
        .sType = v.StructureType.GRAPHICS_PIPELINE_CREATE_INFO,
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

    try checkSuccess(v.vkCreateGraphicsPipelines(
        global_device,
        null,
        @intCast(u32, pipelineInfo.len),
        &pipelineInfo,
        null,
        @as(*[1]v.Pipeline, &graphicsPipeline),
    ));

    v.vkDestroyShaderModule(global_device, fragShaderModule, null);
    v.vkDestroyShaderModule(global_device, vertShaderModule, null);
}

fn createRenderPass() !void {
    const colorAttachment = v.AttachmentDescription{
        .format = swapChainImageFormat,
        .samples = v.SampleCountFlagBits.T_1_BIT,
        .loadOp = v.AttachmentLoadOp.LOAD,
        .storeOp = v.AttachmentStoreOp.STORE,
        .stencilLoadOp = v.AttachmentLoadOp.DONT_CARE,
        .stencilStoreOp = v.AttachmentStoreOp.DONT_CARE,
        .initialLayout = v.ImageLayout.UNDEFINED,
        .finalLayout = v.ImageLayout.PRESENT_SRC_KHR,
        .flags = 0,
    };

    const colorAttachmentRef = v.AttachmentReference{
        .attachment = 0,
        .layout = v.ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
    };

    const subpass = [_]v.SubpassDescription{v.SubpassDescription{
        .pipelineBindPoint = v.PipelineBindPoint.GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = @ptrCast([*]const v.AttachmentReference, &colorAttachmentRef),

        .flags = 0,
        .inputAttachmentCount = 0,
        // .pInputAttachments = null,
        .pResolveAttachments = null,
        .pDepthStencilAttachment = null,
        .preserveAttachmentCount = 0,
        // .pPreserveAttachments = null,
    }};

    const dependency = [_]v.SubpassDependency{v.SubpassDependency{
        .srcSubpass = v.SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = v.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = v.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = v.AccessFlagBits.COLOR_ATTACHMENT_READ_BIT | v.AccessFlagBits.COLOR_ATTACHMENT_WRITE_BIT,

        .dependencyFlags = 0,
    }};

    const renderPassInfo = v.RenderPassCreateInfo{
        .sType = v.StructureType.RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = @ptrCast(*const [1]v.AttachmentDescription, &colorAttachment),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(v.vkCreateRenderPass(global_device, &renderPassInfo, null, &renderPass));
}

fn createImageViews(allocator: *Allocator) !void {
    swapChainImageViews = try allocator.alloc(v.ImageView, swapChainImages.len);
    errdefer allocator.free(swapChainImageViews);

    for (swapChainImages) |swap_chain_image, i| {
        const createInfo = v.ImageViewCreateInfo{
            .sType = v.StructureType.IMAGE_VIEW_CREATE_INFO,
            .image = swap_chain_image,
            .viewType = v.ImageViewType.T_2D,
            .format = swapChainImageFormat,
            .components = v.ComponentMapping{
                .r = v.ComponentSwizzle.IDENTITY,
                .g = v.ComponentSwizzle.IDENTITY,
                .b = v.ComponentSwizzle.IDENTITY,
                .a = v.ComponentSwizzle.IDENTITY,
            },
            .subresourceRange = v.ImageSubresourceRange{
                .aspectMask = v.ImageAspectFlagBits.COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },

            .pNext = null,
            .flags = 0,
        };

        try checkSuccess(v.vkCreateImageView(global_device, &createInfo, null, &swapChainImageViews[i]));
    }
}

fn chooseSwapSurfaceFormat(availableFormats: []v.SurfaceFormatKHR) v.SurfaceFormatKHR {
    if (availableFormats.len == 1 and availableFormats[0].format == v.Format.UNDEFINED) {
        return v.SurfaceFormatKHR{
            .format = v.Format.B8G8R8A8_UNORM,
            .colorSpace = v.ColorSpaceKHR.SRGB_NONLINEAR,
        };
    }

    for (availableFormats) |availableFormat| {
        if (availableFormat.format == v.Format.B8G8R8A8_UNORM and
            availableFormat.colorSpace == v.ColorSpaceKHR.SRGB_NONLINEAR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

fn chooseSwapPresentMode(availablePresentModes: []v.PresentModeKHR) v.PresentModeKHR {
    var bestMode: v.PresentModeKHR = v.PresentModeKHR.FIFO;

    for (availablePresentModes) |availablePresentMode| {
        if (availablePresentMode == v.PresentModeKHR.MAILBOX) {
            return availablePresentMode;
        } else if (availablePresentMode == v.PresentModeKHR.IMMEDIATE) {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

fn chooseSwapExtent(capabilities: v.SurfaceCapabilitiesKHR) v.Extent2D {
    if (capabilities.currentExtent.width != maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        var actualExtent = v.Extent2D{
            .width = WIDTH,
            .height = HEIGHT,
        };

        actualExtent.width = std.math.max(capabilities.minImageExtent.width, std.math.min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std.math.max(capabilities.minImageExtent.height, std.math.min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

fn createSwapChain(allocator: *Allocator) !void {
    var swapChainSupport = try querySwapChainSupport(allocator, physicalDevice);
    defer swapChainSupport.deinit();

    const surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats.toSlice());
    const presentMode = chooseSwapPresentMode(swapChainSupport.presentModes.toSlice());
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

    var createInfo = v.SwapchainCreateInfoKHR{
        .sType = v.StructureType.SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,

        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = v.ImageUsageFlagBits.COLOR_ATTACHMENT_BIT,

        .imageSharingMode = if (different_families) v.SharingMode.CONCURRENT else v.SharingMode.EXCLUSIVE,
        .queueFamilyIndexCount = if (different_families) @as(u32, 2) else @as(u32, 0),
        .pQueueFamilyIndices = if (different_families) &queueFamilyIndices else &([_]u32{ 0, 0 }),

        .preTransform = swapChainSupport.capabilities.currentTransform,
        .compositeAlpha = v.CompositeAlphaFlagBitsKHR.OPAQUE_BIT,
        .presentMode = presentMode,
        .clipped = v.TRUE,

        .oldSwapchain = null,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(v.vkCreateSwapchainKHR(global_device, &createInfo, null, &swapChain));

    try checkSuccess(v.vkGetSwapchainImagesKHR(global_device, swapChain, &imageCount, null));
    swapChainImages = try allocator.alloc(v.Image, imageCount);
    try checkSuccess(v.vkGetSwapchainImagesKHR(global_device, swapChain, &imageCount, swapChainImages.ptr));

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

fn createLogicalDevice(allocator: *Allocator) !void {
    const indices = try findQueueFamilies(allocator, physicalDevice);

    var queueCreateInfos = std.ArrayList(v.DeviceQueueCreateInfo).init(allocator);
    defer queueCreateInfos.deinit();
    const all_queue_families = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };
    const uniqueQueueFamilies = if (indices.graphicsFamily.? == indices.presentFamily.?)
        all_queue_families[0..1]
    else
        all_queue_families[0..2];

    var queuePriority: f32 = 1.0;
    for (uniqueQueueFamilies) |queueFamily| {
        const queueCreateInfo = v.DeviceQueueCreateInfo{
            .sType = v.StructureType.DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = @ptrCast([*]const f32, &queuePriority),
            .pNext = null,
            .flags = 0,
        };
        try queueCreateInfos.append(queueCreateInfo);
    }

    const deviceFeatures = v.PhysicalDeviceFeatures{
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

    const createInfo = v.DeviceCreateInfo{
        .sType = v.StructureType.DEVICE_CREATE_INFO,

        .queueCreateInfoCount = @intCast(u32, queueCreateInfos.len),
        .pQueueCreateInfos = queueCreateInfos.items.ptr,

        .pEnabledFeatures = &deviceFeatures,

        .enabledExtensionCount = @intCast(u32, deviceExtensions.len),
        .ppEnabledExtensionNames = &deviceExtensions,
        .enabledLayerCount = if (enableValidationLayers) @intCast(u32, validationLayers.len) else 0,
        .ppEnabledLayerNames = if (enableValidationLayers) &validationLayers else null,

        .pNext = null,
        .flags = 0,
    };

    try checkSuccess(v.vkCreateDevice(physicalDevice, &createInfo, null, &global_device));

    v.vkGetDeviceQueue(global_device, indices.graphicsFamily.?, 0, &graphicsQueue);
    v.vkGetDeviceQueue(global_device, indices.presentFamily.?, 0, &presentQueue);
}

fn querySwapChainSupport(allocator: *Allocator, device: v.PhysicalDevice) !SwapChainSupportDetails {
    var details = SwapChainSupportDetails.init(allocator);

    try checkSuccess(v.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities));

    var formatCount: u32 = undefined;
    try checkSuccess(v.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, null));

    if (formatCount != 0) {
        try details.formats.resize(formatCount);
        try checkSuccess(v.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.items.ptr));
    }

    var presentModeCount: u32 = undefined;
    try checkSuccess(v.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, null));

    if (presentModeCount != 0) {
        try details.presentModes.resize(presentModeCount);
        try checkSuccess(v.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.items.ptr));
    }

    return details;
}

extern fn debugCallback(
    flags: v.DebugReportFlagsEXT,
    objType: v.DebugReportObjectTypeEXT,
    obj: u64,
    location: usize,
    code: i32,
    layerPrefix: ?[*]const u8,
    msg: ?[*]const u8,
    userData: ?*c_void,
) v.Bool32 {
    std.debug.warn("validation layer: {s}\n", .{@ptrCast([*:0] const u8, msg)});
    return v.FALSE;
}

fn setupDebugCallback() error{FailedToSetUpDebugCallback}!void {
    if (!enableValidationLayers) return;

    var createInfo = v.DebugReportCallbackCreateInfoEXT{
        .sType = v.StructureType.DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
        .flags = v.DebugReportFlagBitsEXT.ERROR_BIT | v.DebugReportFlagBitsEXT.WARNING_BIT,
        .pfnCallback = debugCallback,
        .pNext = null,
        .pUserData = null,
    };

    // if (CreateDebugReportCallbackEXT(&createInfo, null, &callback) != v.Result.SUCCESS) {
    //     return error.FailedToSetUpDebugCallback;
    // }
}

// fn DestroyDebugReportCallbackEXT(
//     pAllocator: ?*const v.AllocationCallbacks,
// ) void {
//     const func = @ptrCast(v.vkDestroyDebugReportCallbackEXT, v.vkGetInstanceProcAddr(
//         instance,
//         "vkDestroyDebugReportCallbackEXT",
//     )) orelse unreachable;
//     func(instance, callback, pAllocator);
// }

fn CreateDebugReportCallbackEXT(
    pCreateInfo: *const v.DebugReportCallbackCreateInfoEXT,
    pAllocator: ?*const v.AllocationCallbacks,
    pCallback: *v.DebugReportCallbackEXT,
) v.Result {
    const func = @ptrCast(v.vkCreateDebugReportCallbackEXT, v.vkGetInstanceProcAddr(
        instance,
        "vkCreateDebugReportCallbackEXT",
    )) orelse return v.Result.ERROR_EXTENSION_NOT_PRESENT;

    return func(instance, pCreateInfo, pAllocator, pCallback);
}

fn checkSuccess(result: v.Result) !void {
    switch (result) {
        v.Result.SUCCESS => {},
        else => return error.Unexpected,
    }
}

fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layerCount: u32 = undefined;

    try checkSuccess(v.vkEnumerateInstanceLayerProperties(&layerCount, null));

    const availableLayers = try allocator.alloc(v.LayerProperties, layerCount);
    defer allocator.free(availableLayers);

    try checkSuccess(v.vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.ptr));

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
    try checkSuccess(v.vkWaitForFences(global_device, 1, @as(*[1]v.Fence, &inFlightFences[currentFrame]), v.TRUE, maxInt(u64)));
    try checkSuccess(v.vkResetFences(global_device, 1, @as(*[1]v.Fence, &inFlightFences[currentFrame])));

    var imageIndex: u32 = undefined;
    try checkSuccess(v.vkAcquireNextImageKHR(global_device, swapChain, maxInt(u64), imageAvailableSemaphores[currentFrame], null, &imageIndex));

    var waitSemaphores = [_]v.Semaphore{imageAvailableSemaphores[currentFrame]};
    var waitStages = [_]v.PipelineStageFlags{v.PipelineStageFlagBits.COLOR_ATTACHMENT_OUTPUT_BIT};

    const signalSemaphores = [_]v.Semaphore{renderFinishedSemaphores[currentFrame]};

    var submitInfo = [_]v.SubmitInfo{v.SubmitInfo{
        .sType = v.StructureType.SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &waitSemaphores,
        .pWaitDstStageMask = &waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = commandBuffers.ptr + imageIndex,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signalSemaphores,

        .pNext = null,
    }};

    try checkSuccess(v.vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));

    const swapChains = [_]v.SwapchainKHR{swapChain};
    const presentInfo = v.PresentInfoKHR{
        .sType = v.StructureType.PRESENT_INFO_KHR,

        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signalSemaphores,

        .swapchainCount = 1,
        .pSwapchains = &swapChains,

        .pImageIndices = @ptrCast(*[1]u32, &imageIndex),

        .pNext = null,
        .pResults = null,
    };

    try checkSuccess(v.vkQueuePresentKHR(presentQueue, &presentInfo));

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