; 64-bit assembly code using AVX instructions that use 512-bit ZMM registers.

.data

;ZERO	dq	8 dup(0)	; 64 bytes of zero
HIMASK	dq 0FFFFFFFFFFFFFFFFh, 0			; Mask to clear upper half of XMM register 
STRIDE	dq 64

.code

sumVector PROC C
	;-----------------------------------------------------------------------------------------------	
	; C prototype: float sumVector(double * inputArray, int array_len)
	; The sumVector proc adds the elements of a passed array. 
	; Parameters
	;   inputArray - address of an array of doubles, passed in RCX
	;   array_len - number of elements of inputArray, passed in RDX
	; Return value
	;   Returns the sum of the elements in the passed array in XMM0.
	; In each loop iteration, zmm1 is set with 8 doubles.  
	;-----------------------------------------------------------------------------------------------
	
; Prologue
	push        rdi			; RDI is a non-volatile register, so save it.
	sub         rsp, 20h  
	mov         rdi, rsp
	push		rsi		; RSI also needs to be saved.

; Initialization
	vzeroall ; Zero out all ZMM registers
	mov rax, rcx			; Copy array address to RAX.
	mov rcx, rdx			; Copy element count to RCX.
	shr rcx, 4			; RCX <- Number of 2 X 8-double chunks.
	xor rsi, rsi			; Zero RSI.			
	
	vorpd xmm3, xmm3, xmmword ptr [HIMASK]  ; Set bits in lower half of XMM3 for later use.

PartialSumsLoop:
	;-----------------------------------------------------------------------------------------------
	; In each iteration, accumulate 8 partial sums of doubles.
	;   When loop terminates, ZMM0 will hold 8 doubles that, 
	;   when added, will be the overall sum of the given array.
	vmovapd zmm1, zmmword ptr [rax + rsi]	; Get 8 doubles and store in ZMM1.
	vaddpd zmm0, zmm0, zmm1	                ; ZMM0 = ZMM0 + ZMM1.				
	add rsi, STRIDE
	vmovapd zmm4, zmmword ptr [rax + rsi]	; Get 8 doubles and store in ZMM4.	
	;add rsi, STRIDE		

	; Accumulate sum from previous loop iteration.
	vaddpd zmm0, zmm0, zmm4	                ; ZMM0 = ZMM0 + ZMM4.
	add rsi, STRIDE			        ; Increment RSI to advance to next block of 8 doubles in the array.
	loop PartialSumsLoop	
	;-----------------------------------------------------------------------------------------------
	
CombineSums:	
	; Use vextractf64x4 to extract 4 doubles (256 bits) from the upper half of ZMM0 to YMM1.
	; YMM0 already contains the lower 256 bits (four doubles) of ZMM0.
	vextractf64x4 ymm1, zmm0, 1	
	vaddpd ymm1, ymm1, ymm0	                  ; YMM0 <-- YMM1 + YMM0.
	; Assuming YMM0 contains y3, y2, y1, y0, and YMM1 contains x3, x2, x1, x0, 
	;   YMM1 now contains as qwords y3+x3, y2+x2, y1+x1, y0+x0
	
	; Use horizontal addition. I'm not aware of any version of vhaddpd for AVX-512.
	;   So I'm limited to working with YMM registers, but not ZMM registers.		
	vhaddpd	ymm1, ymm1, ymm1
	; After vhaddpd, YMM1 now contains as qwords y3+x3+y2+x2 (two times), y1+x1+y0+x0 (two times).

	vmovapd ymm2, ymm1			   ; Copy qwords in YMM1 to YMM2

	; Permute the bits in ymm2, essentially shifting the quadword at index 2 to index 0, 
	;   with all other indexes left unchanged. After permutation, 
	;   YMM2 will contain y3+x3+y2+x2 at index 0, and YMM1 still has y1+x1+y0+x0 at index 0.
	;   Indexes 1, 2, and 3 are don't care.
	vpermpd ymm2, ymm2, 2	
	
	; Add YMM1 and YMM2, storing result in YMM0. We're concerned only with 
	;  the value at index 0.
	vaddpd ymm0, ymm1, ymm2
	
	vandpd xmm0, xmm0, xmm3		          ; Clear upper half of XMM0 for the returned double value.

	
; Epilogue
	pop	rsi
    	add     rsp, 20h		          ; Adjust the stack back to original state
	pop     rdi				  ; Restore RDI	
	ret 
sumVector ENDP

vecProduct PROC C
    ;-----------------------------------------------------------------------------------------------
	; C prototype: void vecProduct(double * outArr, double * inArr1, double * inArr2, int array_len)
	; The vecProduct proc calculates the term-by-term products of the elements
	; of inArr1 and inArr2, and stores the products in outArr. 
	; 
	; Input args: 
	;    outArr - address of output array, passed in RCX.
	;    inArr1 - address of first input array, passed in RDX.
	;    inArr2 - address of second input array, passed in R8.
	;    arr_len - number of elements of each array, passed in R9.
	; Return value: None.
	; On return, RAX holds the address of the vector product.
	; In each iteration, 8 doubles are read from memory and stored in ZMM1, 
	;   and the next 8 doubles are read from memory and stored in ZMM2.
	;-----------------------------------------------------------------------------------------------
	
; Prologue
	push    rdi				; RDI is a non-volatile register, so save it.
	sub     rsp, 20h  
	mov     r10, rsp
	push	rsi				; Save RSI as well.

; Initialization	
	vzeroall				; Zero out all ZMM registers.
	mov rax, rcx			        ; Copy output array address to RAX.
	shr r9, 3				; r9 <- Number of octets of doubles.
	mov rcx, r9				; Copy no. of 8 double chunks to RCX
	xor rsi, rsi			        ; Zero RSI.	
	xor rdi, rdi			        ; Zero RDI.					
			
ProcessChunks:
	;-----------------------------------------------------------------------------------------------
	; Iterate through both arrays, doing term-by-term multiplication.
	vmovapd zmm1, zmmword ptr [rdx + rsi]	; Store 8 doubles from first array to ZMM1.
	vmovapd zmm2, zmmword ptr [r8 + rsi]	; Store 8 doubles from second array to ZMM2.

	; Do term-by-term multiplication of ZMM1 and ZMM2, storing result in ZMM1. 	
	vmulpd zmm0, zmm1, zmm2			; ZMM0 = ZMM1 * ZMM2
	
	; Store products from previous loop iteration into output array memory.
	vmovapd zmmword ptr[rax + rdi], zmm0	
	add rsi, 64			; Advance 8 doubles (64 bytes) higher in first array.
	add rdi, 64			; Advance 8 doubles (64 bytes) higher in secomd array.
	loop ProcessChunks	
	;-----------------------------------------------------------------------------------------------
	
; Epilogue
	pop		rsi		; Restore RSI.
	mov		rsp, r10
    add     rsp, 20h		        ; Adjust the stack back to original state.
	pop     rdi			; Restore RDI.	
	ret 
vecProduct ENDP

end
