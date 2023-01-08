	org 0xF0000000
entry:
	ldl r0, libc
	bind [printf], r0
	mov r0, string
	call printf
	ret

string: data.str "this was printed from C!" data.8 0
printf: data.extern "printf(ptr)"
libc: data.lib "libc.so.6"
