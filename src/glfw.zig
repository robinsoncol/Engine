const vk = @import("vulkan.zig");

pub const GLFWmonitor = @OpaqueType();
pub const GLFWwindow = @OpaqueType();
pub const GLFWcursor = @OpaqueType();

pub const GLFWglproc = ?extern fn () void;
pub const GLFWvkproc = ?extern fn () void;
pub const GLFWerrorfun = ?extern fn (c_int, ?[*]const u8) void;
pub const GLFWwindowposfun = ?extern fn (?*GLFWwindow, c_int, c_int) void;
pub const GLFWwindowsizefun = ?extern fn (?*GLFWwindow, c_int, c_int) void;
pub const GLFWwindowclosefun = ?extern fn (?*GLFWwindow) void;
pub const GLFWwindowrefreshfun = ?extern fn (?*GLFWwindow) void;
pub const GLFWwindowfocusfun = ?extern fn (?*GLFWwindow, c_int) void;
pub const GLFWwindowiconifyfun = ?extern fn (?*GLFWwindow, c_int) void;
pub const GLFWframebuffersizefun = ?extern fn (?*GLFWwindow, c_int, c_int) void;
pub const GLFWmousebuttonfun = ?extern fn (?*GLFWwindow, c_int, c_int, c_int) void;
pub const GLFWcursorposfun = ?extern fn (?*GLFWwindow, f64, f64) void;
pub const GLFWcursorenterfun = ?extern fn (?*GLFWwindow, c_int) void;
pub const GLFWscrollfun = ?extern fn (?*GLFWwindow, f64, f64) void;
pub const GLFWkeyfun = ?extern fn (?*GLFWwindow, c_int, c_int, c_int, c_int) void;
pub const GLFWcharfun = ?extern fn (?*GLFWwindow, c_uint) void;
pub const GLFWcharmodsfun = ?extern fn (?*GLFWwindow, c_uint, c_int) void;
pub const GLFWdropfun = ?extern fn (?*GLFWwindow, c_int, ?[*](?[*]const u8)) void;
pub const GLFWmonitorfun = ?extern fn (?*GLFWmonitor, c_int) void;
pub const GLFWjoystickfun = ?extern fn (c_int, c_int) void;

pub const GLFWvidmode = extern struct {
    width: c_int,
    height: c_int,
    redBits: c_int,
    greenBits: c_int,
    blueBits: c_int,
    refreshRate: c_int,
};

pub const GLFWgammaramp = extern struct {
    red: ?[*]c_ushort,
    green: ?[*]c_ushort,
    blue: ?[*]c_ushort,
    size: c_uint,
};

pub const GLFWimage = extern struct {
    width: c_int,
    height: c_int,
    pixels: ?[*]u8,
};

pub extern fn glfwInit() c_int;
pub extern fn glfwTerminate() void;
pub extern fn glfwGetVersion(major: ?[*]c_int, minor: ?[*]c_int, rev: ?[*]c_int) void;
pub extern fn glfwGetVersionString() ?[*]const u8;
pub extern fn glfwSetErrorCallback(cbfun: GLFWerrorfun) GLFWerrorfun;
pub extern fn glfwGetMonitors(count: ?[*]c_int) ?[*](?*GLFWmonitor);
pub extern fn glfwGetPrimaryMonitor() ?*GLFWmonitor;
pub extern fn glfwGetMonitorPos(monitor: ?*GLFWmonitor, xpos: ?[*]c_int, ypos: ?[*]c_int) void;
pub extern fn glfwGetMonitorPhysicalSize(monitor: ?*GLFWmonitor, widthMM: ?[*]c_int, heightMM: ?[*]c_int) void;
pub extern fn glfwGetMonitorName(monitor: ?*GLFWmonitor) ?[*]const u8;
pub extern fn glfwSetMonitorCallback(cbfun: GLFWmonitorfun) GLFWmonitorfun;
pub extern fn glfwGetVideoModes(monitor: ?*GLFWmonitor, count: ?[*]c_int) ?[*]const GLFWvidmode;
pub extern fn glfwGetVideoMode(monitor: ?*GLFWmonitor) ?[*]const GLFWvidmode;
pub extern fn glfwSetGamma(monitor: ?*GLFWmonitor, gamma: f32) void;
pub extern fn glfwGetGammaRamp(monitor: ?*GLFWmonitor) ?[*]const GLFWgammaramp;
pub extern fn glfwSetGammaRamp(monitor: ?*GLFWmonitor, ramp: ?[*]const GLFWgammaramp) void;
pub extern fn glfwDefaultWindowHints() void;
pub extern fn glfwWindowHint(hint: c_int, value: c_int) void;
pub extern fn glfwCreateWindow(width: c_int, height: c_int, title: ?[*]const u8, monitor: ?*GLFWmonitor, share: ?*GLFWwindow) ?*GLFWwindow;
pub extern fn glfwDestroyWindow(window: ?*GLFWwindow) void;
pub extern fn glfwWindowShouldClose(window: ?*GLFWwindow) c_int;
pub extern fn glfwSetWindowShouldClose(window: ?*GLFWwindow, value: c_int) void;
pub extern fn glfwSetWindowTitle(window: ?*GLFWwindow, title: ?[*]const u8) void;
pub extern fn glfwSetWindowIcon(window: ?*GLFWwindow, count: c_int, images: ?[*]const GLFWimage) void;
pub extern fn glfwGetWindowPos(window: ?*GLFWwindow, xpos: ?[*]c_int, ypos: ?[*]c_int) void;
pub extern fn glfwSetWindowPos(window: ?*GLFWwindow, xpos: c_int, ypos: c_int) void;
pub extern fn glfwGetWindowSize(window: ?*GLFWwindow, width: ?[*]c_int, height: ?[*]c_int) void;
pub extern fn glfwSetWindowSizeLimits(window: ?*GLFWwindow, minwidth: c_int, minheight: c_int, maxwidth: c_int, maxheight: c_int) void;
pub extern fn glfwSetWindowAspectRatio(window: ?*GLFWwindow, numer: c_int, denom: c_int) void;
pub extern fn glfwSetWindowSize(window: ?*GLFWwindow, width: c_int, height: c_int) void;
pub extern fn glfwGetFramebufferSize(window: ?*GLFWwindow, width: ?[*]c_int, height: ?[*]c_int) void;
pub extern fn glfwGetWindowFrameSize(window: ?*GLFWwindow, left: ?[*]c_int, top: ?[*]c_int, right: ?[*]c_int, bottom: ?[*]c_int) void;
pub extern fn glfwIconifyWindow(window: ?*GLFWwindow) void;
pub extern fn glfwRestoreWindow(window: ?*GLFWwindow) void;
pub extern fn glfwMaximizeWindow(window: ?*GLFWwindow) void;
pub extern fn glfwShowWindow(window: ?*GLFWwindow) void;
pub extern fn glfwHideWindow(window: ?*GLFWwindow) void;
pub extern fn glfwFocusWindow(window: ?*GLFWwindow) void;
pub extern fn glfwGetWindowMonitor(window: ?*GLFWwindow) ?*GLFWmonitor;
pub extern fn glfwSetWindowMonitor(window: ?*GLFWwindow, monitor: ?*GLFWmonitor, xpos: c_int, ypos: c_int, width: c_int, height: c_int, refreshRate: c_int) void;
pub extern fn glfwGetWindowAttrib(window: ?*GLFWwindow, attrib: c_int) c_int;
pub extern fn glfwSetWindowUserPointer(window: ?*GLFWwindow, pointer: ?*c_void) void;
pub extern fn glfwGetWindowUserPointer(window: ?*GLFWwindow) ?*c_void;
pub extern fn glfwSetWindowPosCallback(window: ?*GLFWwindow, cbfun: GLFWwindowposfun) GLFWwindowposfun;
pub extern fn glfwSetWindowSizeCallback(window: ?*GLFWwindow, cbfun: GLFWwindowsizefun) GLFWwindowsizefun;
pub extern fn glfwSetWindowCloseCallback(window: ?*GLFWwindow, cbfun: GLFWwindowclosefun) GLFWwindowclosefun;
pub extern fn glfwSetWindowRefreshCallback(window: ?*GLFWwindow, cbfun: GLFWwindowrefreshfun) GLFWwindowrefreshfun;
pub extern fn glfwSetWindowFocusCallback(window: ?*GLFWwindow, cbfun: GLFWwindowfocusfun) GLFWwindowfocusfun;
pub extern fn glfwSetWindowIconifyCallback(window: ?*GLFWwindow, cbfun: GLFWwindowiconifyfun) GLFWwindowiconifyfun;
pub extern fn glfwSetFramebufferSizeCallback(window: ?*GLFWwindow, cbfun: GLFWframebuffersizefun) GLFWframebuffersizefun;
pub extern fn glfwPollEvents() void;
pub extern fn glfwWaitEvents() void;
pub extern fn glfwWaitEventsTimeout(timeout: f64) void;
pub extern fn glfwPostEmptyEvent() void;
pub extern fn glfwGetInputMode(window: ?*GLFWwindow, mode: c_int) c_int;
pub extern fn glfwSetInputMode(window: ?*GLFWwindow, mode: c_int, value: c_int) void;
pub extern fn glfwGetKeyName(key: c_int, scancode: c_int) ?[*]const u8;
pub extern fn glfwGetKey(window: ?*GLFWwindow, key: c_int) c_int;
pub extern fn glfwGetMouseButton(window: ?*GLFWwindow, button: c_int) c_int;
pub extern fn glfwGetCursorPos(window: ?*GLFWwindow, xpos: ?[*]f64, ypos: ?[*]f64) void;
pub extern fn glfwSetCursorPos(window: ?*GLFWwindow, xpos: f64, ypos: f64) void;
pub extern fn glfwCreateCursor(image: ?[*]const GLFWimage, xhot: c_int, yhot: c_int) ?*GLFWcursor;
pub extern fn glfwCreateStandardCursor(shape: c_int) ?*GLFWcursor;
pub extern fn glfwDestroyCursor(cursor: ?*GLFWcursor) void;
pub extern fn glfwSetCursor(window: ?*GLFWwindow, cursor: ?*GLFWcursor) void;
pub extern fn glfwSetKeyCallback(window: ?*GLFWwindow, cbfun: GLFWkeyfun) GLFWkeyfun;
pub extern fn glfwSetCharCallback(window: ?*GLFWwindow, cbfun: GLFWcharfun) GLFWcharfun;
pub extern fn glfwSetCharModsCallback(window: ?*GLFWwindow, cbfun: GLFWcharmodsfun) GLFWcharmodsfun;
pub extern fn glfwSetMouseButtonCallback(window: ?*GLFWwindow, cbfun: GLFWmousebuttonfun) GLFWmousebuttonfun;
pub extern fn glfwSetCursorPosCallback(window: ?*GLFWwindow, cbfun: GLFWcursorposfun) GLFWcursorposfun;
pub extern fn glfwSetCursorEnterCallback(window: ?*GLFWwindow, cbfun: GLFWcursorenterfun) GLFWcursorenterfun;
pub extern fn glfwSetScrollCallback(window: ?*GLFWwindow, cbfun: GLFWscrollfun) GLFWscrollfun;
pub extern fn glfwSetDropCallback(window: ?*GLFWwindow, cbfun: GLFWdropfun) GLFWdropfun;
pub extern fn glfwJoystickPresent(joy: c_int) c_int;
pub extern fn glfwGetJoystickAxes(joy: c_int, count: ?[*]c_int) ?[*]const f32;
pub extern fn glfwGetJoystickButtons(joy: c_int, count: ?[*]c_int) ?[*]const u8;
pub extern fn glfwGetJoystickName(joy: c_int) ?[*]const u8;
pub extern fn glfwSetJoystickCallback(cbfun: GLFWjoystickfun) GLFWjoystickfun;
pub extern fn glfwSetClipboardString(window: ?*GLFWwindow, string: ?[*]const u8) void;
pub extern fn glfwGetClipboardString(window: ?*GLFWwindow) ?[*]const u8;
pub extern fn glfwGetTime() f64;
pub extern fn glfwSetTime(time: f64) void;
pub extern fn glfwGetTimerValue() u64;
pub extern fn glfwGetTimerFrequency() u64;
pub extern fn glfwMakeContextCurrent(window: ?*GLFWwindow) void;
pub extern fn glfwGetCurrentContext() ?*GLFWwindow;
pub extern fn glfwSwapBuffers(window: ?*GLFWwindow) void;
pub extern fn glfwSwapInterval(interval: c_int) void;
pub extern fn glfwExtensionSupported(extension: ?[*]const u8) c_int;
pub extern fn glfwGetProcAddress(procname: ?[*]const u8) GLFWglproc;
pub extern fn glfwVulkanSupported() c_int;
pub extern fn glfwGetRequiredInstanceExtensions(count: *u32) [*]const [*]const u8;
pub extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: ?[*]const u8) GLFWvkproc;
pub extern fn glfwGetPhysicalDevicePresentationSupport(instance: vk.Instance, device: vk.PhysicalDevice, queuefamily: u32) c_int;
pub extern fn glfwCreateWindowSurface(instance: vk.Instance, window: *GLFWwindow, allocator: ?*const vk.AllocationCallbacks, surface: *vk.SurfaceKHR) vk.Result;

pub const GLFW_ACCUM_ALPHA_BITS = 135178;
pub const GLFW_ACCUM_BLUE_BITS = 135177;
pub const GLFW_ACCUM_GREEN_BITS = 135176;
pub const GLFW_ACCUM_RED_BITS = 135175;
pub const GLFW_ALPHA_BITS = 135172;
pub const GLFW_ANY_RELEASE_BEHAVIOR = 0;
pub const GLFW_API_UNAVAILABLE = 65542;
pub const GLFW_ARROW_CURSOR = 221185;
pub const GLFW_AUTO_ICONIFY = 131078;
pub const GLFW_AUX_BUFFERS = 135179;
pub const GLFW_BLUE_BITS = 135171;
pub const GLFW_CLIENT_API = 139265;
pub const GLFW_CONNECTED = 262145;
pub const GLFW_CONTEXT_CREATION_API = 139275;
pub const GLFW_CONTEXT_NO_ERROR = 139274;
pub const GLFW_CONTEXT_RELEASE_BEHAVIOR = 139273;
pub const GLFW_CONTEXT_REVISION = 139268;
pub const GLFW_CONTEXT_ROBUSTNESS = 139269;
pub const GLFW_CONTEXT_VERSION_MAJOR = 139266;
pub const GLFW_CONTEXT_VERSION_MINOR = 139267;
pub const GLFW_CROSSHAIR_CURSOR = 221187;
pub const GLFW_CURSOR = 208897;
pub const GLFW_CURSOR_DISABLED = 212995;
pub const GLFW_CURSOR_HIDDEN = 212994;
pub const GLFW_CURSOR_NORMAL = 212993;
pub const GLFW_DECORATED = 131077;
pub const GLFW_DEPTH_BITS = 135173;
pub const GLFW_DISCONNECTED = 262146;
pub const GLFW_DONT_CARE = -1;
pub const GLFW_DOUBLEBUFFER = 135184;
pub const GLFW_EGL_CONTEXT_API = 221186;
pub const GLFW_FALSE = 0;
pub const GLFW_FLOATING = 131079;
pub const GLFW_FOCUSED = 131073;
pub const GLFW_FORMAT_UNAVAILABLE = 65545;
pub const GLFW_GREEN_BITS = 135170;
pub const GLFW_HAND_CURSOR = 221188;
pub const GLFW_HRESIZE_CURSOR = 221189;
pub const GLFW_IBEAM_CURSOR = 221186;
pub const GLFW_ICONIFIED = 131074;
pub const GLFW_INVALID_ENUM = 65539;
pub const GLFW_INVALID_VALUE = 65540;
pub const GLFW_JOYSTICK_1 = 0;
pub const GLFW_JOYSTICK_10 = 9;
pub const GLFW_JOYSTICK_11 = 10;
pub const GLFW_JOYSTICK_12 = 11;
pub const GLFW_JOYSTICK_13 = 12;
pub const GLFW_JOYSTICK_14 = 13;
pub const GLFW_JOYSTICK_15 = 14;
pub const GLFW_JOYSTICK_16 = 15;
pub const GLFW_JOYSTICK_2 = 1;
pub const GLFW_JOYSTICK_3 = 2;
pub const GLFW_JOYSTICK_4 = 3;
pub const GLFW_JOYSTICK_5 = 4;
pub const GLFW_JOYSTICK_6 = 5;
pub const GLFW_JOYSTICK_7 = 6;
pub const GLFW_JOYSTICK_8 = 7;
pub const GLFW_JOYSTICK_9 = 8;
pub const GLFW_JOYSTICK_LAST = GLFW_JOYSTICK_16;
pub const GLFW_KEY_0 = 48;
pub const GLFW_KEY_1 = 49;
pub const GLFW_KEY_2 = 50;
pub const GLFW_KEY_3 = 51;
pub const GLFW_KEY_4 = 52;
pub const GLFW_KEY_5 = 53;
pub const GLFW_KEY_6 = 54;
pub const GLFW_KEY_7 = 55;
pub const GLFW_KEY_8 = 56;
pub const GLFW_KEY_9 = 57;
pub const GLFW_KEY_A = 65;
pub const GLFW_KEY_APOSTROPHE = 39;
pub const GLFW_KEY_B = 66;
pub const GLFW_KEY_BACKSLASH = 92;
pub const GLFW_KEY_BACKSPACE = 259;
pub const GLFW_KEY_C = 67;
pub const GLFW_KEY_CAPS_LOCK = 280;
pub const GLFW_KEY_COMMA = 44;
pub const GLFW_KEY_D = 68;
pub const GLFW_KEY_DELETE = 261;
pub const GLFW_KEY_DOWN = 264;
pub const GLFW_KEY_E = 69;
pub const GLFW_KEY_END = 269;
pub const GLFW_KEY_ENTER = 257;
pub const GLFW_KEY_EQUAL = 61;
pub const GLFW_KEY_ESCAPE = 256;
pub const GLFW_KEY_F = 70;
pub const GLFW_KEY_F1 = 290;
pub const GLFW_KEY_F10 = 299;
pub const GLFW_KEY_F11 = 300;
pub const GLFW_KEY_F12 = 301;
pub const GLFW_KEY_F13 = 302;
pub const GLFW_KEY_F14 = 303;
pub const GLFW_KEY_F15 = 304;
pub const GLFW_KEY_F16 = 305;
pub const GLFW_KEY_F17 = 306;
pub const GLFW_KEY_F18 = 307;
pub const GLFW_KEY_F19 = 308;
pub const GLFW_KEY_F2 = 291;
pub const GLFW_KEY_F20 = 309;
pub const GLFW_KEY_F21 = 310;
pub const GLFW_KEY_F22 = 311;
pub const GLFW_KEY_F23 = 312;
pub const GLFW_KEY_F24 = 313;
pub const GLFW_KEY_F25 = 314;
pub const GLFW_KEY_F3 = 292;
pub const GLFW_KEY_F4 = 293;
pub const GLFW_KEY_F5 = 294;
pub const GLFW_KEY_F6 = 295;
pub const GLFW_KEY_F7 = 296;
pub const GLFW_KEY_F8 = 297;
pub const GLFW_KEY_F9 = 298;
pub const GLFW_KEY_G = 71;
pub const GLFW_KEY_GRAVE_ACCENT = 96;
pub const GLFW_KEY_H = 72;
pub const GLFW_KEY_HOME = 268;
pub const GLFW_KEY_I = 73;
pub const GLFW_KEY_INSERT = 260;
pub const GLFW_KEY_J = 74;
pub const GLFW_KEY_K = 75;
pub const GLFW_KEY_KP_0 = 320;
pub const GLFW_KEY_KP_1 = 321;
pub const GLFW_KEY_KP_2 = 322;
pub const GLFW_KEY_KP_3 = 323;
pub const GLFW_KEY_KP_4 = 324;
pub const GLFW_KEY_KP_5 = 325;
pub const GLFW_KEY_KP_6 = 326;
pub const GLFW_KEY_KP_7 = 327;
pub const GLFW_KEY_KP_8 = 328;
pub const GLFW_KEY_KP_9 = 329;
pub const GLFW_KEY_KP_ADD = 334;
pub const GLFW_KEY_KP_DECIMAL = 330;
pub const GLFW_KEY_KP_DIVIDE = 331;
pub const GLFW_KEY_KP_ENTER = 335;
pub const GLFW_KEY_KP_EQUAL = 336;
pub const GLFW_KEY_KP_MULTIPLY = 332;
pub const GLFW_KEY_KP_SUBTRACT = 333;
pub const GLFW_KEY_L = 76;
pub const GLFW_KEY_LAST = GLFW_KEY_MENU;
pub const GLFW_KEY_LEFT = 263;
pub const GLFW_KEY_LEFT_ALT = 342;
pub const GLFW_KEY_LEFT_BRACKET = 91;
pub const GLFW_KEY_LEFT_CONTROL = 341;
pub const GLFW_KEY_LEFT_SHIFT = 340;
pub const GLFW_KEY_LEFT_SUPER = 343;
pub const GLFW_KEY_M = 77;
pub const GLFW_KEY_MENU = 348;
pub const GLFW_KEY_MINUS = 45;
pub const GLFW_KEY_N = 78;
pub const GLFW_KEY_NUM_LOCK = 282;
pub const GLFW_KEY_O = 79;
pub const GLFW_KEY_P = 80;
pub const GLFW_KEY_PAGE_DOWN = 267;
pub const GLFW_KEY_PAGE_UP = 266;
pub const GLFW_KEY_PAUSE = 284;
pub const GLFW_KEY_PERIOD = 46;
pub const GLFW_KEY_PRINT_SCREEN = 283;
pub const GLFW_KEY_Q = 81;
pub const GLFW_KEY_R = 82;
pub const GLFW_KEY_RIGHT = 262;
pub const GLFW_KEY_RIGHT_ALT = 346;
pub const GLFW_KEY_RIGHT_BRACKET = 93;
pub const GLFW_KEY_RIGHT_CONTROL = 345;
pub const GLFW_KEY_RIGHT_SHIFT = 344;
pub const GLFW_KEY_RIGHT_SUPER = 347;
pub const GLFW_KEY_S = 83;
pub const GLFW_KEY_SCROLL_LOCK = 281;
pub const GLFW_KEY_SEMICOLON = 59;
pub const GLFW_KEY_SLASH = 47;
pub const GLFW_KEY_SPACE = 32;
pub const GLFW_KEY_T = 84;
pub const GLFW_KEY_TAB = 258;
pub const GLFW_KEY_U = 85;
pub const GLFW_KEY_UNKNOWN = -1;
pub const GLFW_KEY_UP = 265;
pub const GLFW_KEY_V = 86;
pub const GLFW_KEY_W = 87;
pub const GLFW_KEY_WORLD_1 = 161;
pub const GLFW_KEY_WORLD_2 = 162;
pub const GLFW_KEY_X = 88;
pub const GLFW_KEY_Y = 89;
pub const GLFW_KEY_Z = 90;
pub const GLFW_LOSE_CONTEXT_ON_RESET = 200706;
pub const GLFW_MAXIMIZED = 131080;
pub const GLFW_MOD_ALT = 4;
pub const GLFW_MOD_CONTROL = 2;
pub const GLFW_MOD_SHIFT = 1;
pub const GLFW_MOD_SUPER = 8;
pub const GLFW_MOUSE_BUTTON_1 = 0;
pub const GLFW_MOUSE_BUTTON_2 = 1;
pub const GLFW_MOUSE_BUTTON_3 = 2;
pub const GLFW_MOUSE_BUTTON_4 = 3;
pub const GLFW_MOUSE_BUTTON_5 = 4;
pub const GLFW_MOUSE_BUTTON_6 = 5;
pub const GLFW_MOUSE_BUTTON_7 = 6;
pub const GLFW_MOUSE_BUTTON_8 = 7;
pub const GLFW_MOUSE_BUTTON_LAST = GLFW_MOUSE_BUTTON_8;
pub const GLFW_MOUSE_BUTTON_LEFT = GLFW_MOUSE_BUTTON_1;
pub const GLFW_MOUSE_BUTTON_MIDDLE = GLFW_MOUSE_BUTTON_3;
pub const GLFW_MOUSE_BUTTON_RIGHT = GLFW_MOUSE_BUTTON_2;
pub const GLFW_NATIVE_CONTEXT_API = 221185;
pub const GLFW_NOT_INITIALIZED = 65537;
pub const GLFW_NO_API = 0;
pub const GLFW_NO_CURRENT_CONTEXT = 65538;
pub const GLFW_NO_RESET_NOTIFICATION = 200705;
pub const GLFW_NO_ROBUSTNESS = 0;
pub const GLFW_NO_WINDOW_CONTEXT = 65546;
pub const GLFW_OPENGL_ANY_PROFILE = 0;
pub const GLFW_OPENGL_API = 196609;
pub const GLFW_OPENGL_COMPAT_PROFILE = 204802;
pub const GLFW_OPENGL_CORE_PROFILE = 204801;
pub const GLFW_OPENGL_DEBUG_CONTEXT = 139271;
pub const GLFW_OPENGL_ES_API = 196610;
pub const GLFW_OPENGL_FORWARD_COMPAT = 139270;
pub const GLFW_OPENGL_PROFILE = 139272;
pub const GLFW_OUT_OF_MEMORY = 65541;
pub const GLFW_PLATFORM_ERROR = 65544;
pub const GLFW_PRESS = 1;
pub const GLFW_RED_BITS = 135169;
pub const GLFW_REFRESH_RATE = 135183;
pub const GLFW_RELEASE = 0;
pub const GLFW_RELEASE_BEHAVIOR_FLUSH = 217089;
pub const GLFW_RELEASE_BEHAVIOR_NONE = 217090;
pub const GLFW_REPEAT = 2;
pub const GLFW_RESIZABLE = 131075;
pub const GLFW_SAMPLES = 135181;
pub const GLFW_SRGB_CAPABLE = 135182;
pub const GLFW_STENCIL_BITS = 135174;
pub const GLFW_STEREO = 135180;
pub const GLFW_STICKY_KEYS = 208898;
pub const GLFW_STICKY_MOUSE_BUTTONS = 208899;
pub const GLFW_TRUE = 1;
pub const GLFW_VERSION_MAJOR = 3;
pub const GLFW_VERSION_MINOR = 2;
pub const GLFW_VERSION_REVISION = 1;
pub const GLFW_VERSION_UNAVAILABLE = 65543;
pub const GLFW_VISIBLE = 131076;
pub const GLFW_VRESIZE_CURSOR = 221190;