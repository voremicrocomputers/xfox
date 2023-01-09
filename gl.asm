	org 0xF0000000
entry:
	ldl r20, glfw
	bind [glfwInit], r20
	bind [glfwTerminate], r20
	bind [glfwCreateWindow], r20
	bind [glfwWindowShouldClose], r20
	bind [glfwMakeContextCurrent], r20
	bind [glfwSwapBuffers], r20

	ldl r20, gl
	bind [glClear], r20
	call glfwInit
	mov r0, 640
	mov r1, 480
	mov r2, title
	mov r3, 0
	mov r4, 0
	call glfwCreateWindow
	mov r4, r0
	rse
	call glfwMakeContextCurrent
	rcl
	mov r0, GL_COLOR_BUFFER_BIT
	call glClear
	mov r0, r4
	rse
	call glfwSwapBuffers
	rcl
loop_forever:
	nop
	jmp loop_forever
	ret

; strings
title: data.str "fox32!" data.8 0

; constants
GL_COLOR_BUFFER_BIT: data.32 16384

; external functions
glfwInit: data.extern "glfwInit()"
glfwTerminate: data.extern "glfwTerminate()"
glfwCreateWindow: data.extern "glfwCreateWindow(u32,u32,ptr,ptr,ptr)"
glfwWindowShouldClose: data.extern "glfwWindowShouldClose(ptr)"
glfwMakeContextCurrent: data.extern "glfwMakeContextCurrent(ptr)"
glfwSwapBuffers: data.extern "glfwSwapBuffers(ptr)"

glClear: data.extern "glClear(u32)"

; libraries
glfw: data.lib "libglfw"
gl: data.lib "libGL"

