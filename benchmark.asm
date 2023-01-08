; incredibly simple benchmark:
; set r0 to 0x0
; add 4 to r0
; divide by 2 on r0
; loop that 1,000,000 times
; check how long execution took
	org 0xF0000000

entry:
	mov rsp, 0x01FFF800
	mov r0, string
	call debug_print	
	mov r2, 1000000
loop:
	mov r3, 0
	add r3, 4
	div r3, 2
	add r1, 1
	cmp r1, r2
	ifz jmp loop_end
	jmp loop

loop_end:
	mov r0, string_end
	call debug_print
	ret

debug_print:
	push r0
	push r1
debug_print_loop:
	mov r1, 0x00000000
	out r1, [r0]
	inc r0
	cmp.8 [r0], 0x00
	ifnz jmp debug_print_loop
	pop r1
	pop r0
	ret


string: data.str "start!" data.8 0
string_end: data.str "end!" data.8 0
string_going: data.str "going!" data.8 0
