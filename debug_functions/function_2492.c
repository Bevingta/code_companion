static inline long div_ll_X_l_rem(long long divs, long div, long *rem)
{
	long dum2;
	asm("divl %2":"=a"(dum2), "=d"(*rem)
	    : "rm"(div), "A"(divs));

	return dum2;

}