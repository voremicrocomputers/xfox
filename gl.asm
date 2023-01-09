	org 0xF0000000
entry:
	ldl r20, glfw
	bind [glfwInit], r20
	bind [glfwTerminate], r20
	bind [glfwCreateWindow], r20
	bind [glfwWindowShouldClose], r20
	call glfwInit
	mov r0, 640
	mov r1, 480
	mov r2, title
	mov r3, 0
	mov r4, 0
	call glfwCreateWindow
loop_forever:
	nop
	jmp loop_forever
	ret

; strings
title: data.str "fox32!" data.8 0

; external functions
glfwInit: data.extern "glfwInit()"
glfwTerminate: data.extern "glfwTerminate()"
glfwCreateWindow: data.extern "glfwCreateWindow(u32,u32,ptr,ptr,ptr)"
glfwWindowShouldClose: data.extern "glfwWindowShouldClose(ptr)"

; libraries
glfw: data.lib "libglfw"

