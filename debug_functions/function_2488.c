fbComposite (CARD8      op,
	     PicturePtr pSrc,
	     PicturePtr pMask,
	     PicturePtr pDst,
	     INT16      xSrc,
	     INT16      ySrc,
	     INT16      xMask,
	     INT16      yMask,
	     INT16      xDst,
	     INT16      yDst,
	     CARD16     width,
	     CARD16     height)
{
    RegionRec	    region;
    int		    n;
    BoxPtr	    pbox;
    CompositeFunc   func = NULL;
    Bool	    srcRepeat = pSrc->pDrawable && pSrc->repeatType == RepeatNormal;
    Bool	    maskRepeat = FALSE;
    Bool	    srcTransform = pSrc->transform != 0;
    Bool	    maskTransform = FALSE;
    Bool	    srcAlphaMap = pSrc->alphaMap != 0;
    Bool	    maskAlphaMap = FALSE;
    Bool	    dstAlphaMap = pDst->alphaMap != 0;
    int		    x_msk, y_msk, x_src, y_src, x_dst, y_dst;
    int		    w, h, w_this, h_this;

#ifdef USE_MMX
    static Bool mmx_setup = FALSE;
    if (!mmx_setup) {
        fbComposeSetupMMX();
        mmx_setup = TRUE;
    }
#endif

    xDst += pDst->pDrawable->x;
    yDst += pDst->pDrawable->y;
    if (pSrc->pDrawable) {
        xSrc += pSrc->pDrawable->x;
        ySrc += pSrc->pDrawable->y;
    }

    if (srcRepeat && srcTransform &&
	pSrc->pDrawable->width == 1 &&
	pSrc->pDrawable->height == 1)
	srcTransform = FALSE;
    
    if (pMask && pMask->pDrawable)
    {
	xMask += pMask->pDrawable->x;
	yMask += pMask->pDrawable->y;
	maskRepeat = pMask->repeatType == RepeatNormal;

	if (pMask->filter == PictFilterConvolution)
	    maskTransform = TRUE;

	maskAlphaMap = pMask->alphaMap != 0;

	if (maskRepeat && maskTransform &&
	    pMask->pDrawable->width == 1 &&
	    pMask->pDrawable->height == 1)
	    maskTransform = FALSE;
    }

    if (pSrc->pDrawable && (!pMask || pMask->pDrawable)
        && !srcTransform && !maskTransform
        && !maskAlphaMap && !srcAlphaMap && !dstAlphaMap
        && (pSrc->filter != PictFilterConvolution)
        && (!pMask || pMask->filter != PictFilterConvolution))
    switch (op) {
    case PictOpOver:
	if (pMask)
	{
	    if (fbCanGetSolid(pSrc) &&
		!maskRepeat)
	    {
		if (PICT_FORMAT_COLOR(pSrc->format)) {
		    switch (pMask->format) {
		    case PICT_a8:
			switch (pDst->format) {
			case PICT_r5g6b5:
			case PICT_b5g6r5:
#ifdef USE_MMX
			    if (fbHaveMMX())
				func = fbCompositeSolidMask_nx8x0565mmx;
			    else
#endif
				func = fbCompositeSolidMask_nx8x0565;
			    break;
			case PICT_r8g8b8:
			case PICT_b8g8r8:
			    func = fbCompositeSolidMask_nx8x0888;
			    break;
			case PICT_a8r8g8b8:
			case PICT_x8r8g8b8:
			case PICT_a8b8g8r8:
			case PICT_x8b8g8r8:
#ifdef USE_MMX
			    if (fbHaveMMX())
				func = fbCompositeSolidMask_nx8x8888mmx;
			    else
#endif
				func = fbCompositeSolidMask_nx8x8888;
			    break;
			default:
			    break;
			}
			break;
		    case PICT_a8r8g8b8:
			if (pMask->componentAlpha) {
			    switch (pDst->format) {
			    case PICT_a8r8g8b8:
			    case PICT_x8r8g8b8:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSolidMask_nx8888x8888Cmmx;
				else
#endif
				    func = fbCompositeSolidMask_nx8888x8888C;
				break;
			    case PICT_r5g6b5:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSolidMask_nx8888x0565Cmmx;
				else
#endif
				    func = fbCompositeSolidMask_nx8888x0565C;
				break;
			    default:
				break;
			    }
			}
			else
			{
			    switch (pDst->format) {
                            case PICT_r5g6b5:
                                func = fbCompositeSolidMask_nx8888x0565;
                                break;
			    default:
				break;
                            }
			}
			break;
		    case PICT_a8b8g8r8:
			if (pMask->componentAlpha) {
			    switch (pDst->format) {
			    case PICT_a8b8g8r8:
			    case PICT_x8b8g8r8:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSolidMask_nx8888x8888Cmmx;
				else
#endif
				    func = fbCompositeSolidMask_nx8888x8888C;
				break;
			    case PICT_b5g6r5:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSolidMask_nx8888x0565Cmmx;
				else
#endif
				    func = fbCompositeSolidMask_nx8888x0565C;
				break;
			    default:
				break;
			    }
			}
			else
			{
			    switch (pDst->format) {
                            case PICT_b5g6r5:
                                func = fbCompositeSolidMask_nx8888x0565;
                                break;
			    default:
				break;
                            }
			}
			break;
		    case PICT_a1:
			switch (pDst->format) {
			case PICT_r5g6b5:
			case PICT_b5g6r5:
			case PICT_r8g8b8:
			case PICT_b8g8r8:
			case PICT_a8r8g8b8:
			case PICT_x8r8g8b8:
			case PICT_a8b8g8r8:
			case PICT_x8b8g8r8:
			{
			    FbBits src;

			    fbComposeGetSolid(pSrc, src, pDst->format);
			    if ((src & 0xff000000) == 0xff000000)
				func = fbCompositeSolidMask_nx1xn;
			    break;
			}
			default:
			    break;
			}
			break;
		    default:
			break;
		    }
		}
		if (func)
		    srcRepeat = FALSE;
	    }
	    else if (!srcRepeat) /* has mask and non-repeating source */
	    {
		if (pSrc->pDrawable == pMask->pDrawable &&
		    xSrc == xMask && ySrc == yMask &&
		    !pMask->componentAlpha && !maskRepeat)
		{
		    /* source == mask: non-premultiplied data */
		    switch (pSrc->format) {
		    case PICT_x8b8g8r8:
			switch (pMask->format) {
			case PICT_a8r8g8b8:
			case PICT_a8b8g8r8:
			    switch (pDst->format) {
			    case PICT_a8r8g8b8:
			    case PICT_x8r8g8b8:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSrc_8888RevNPx8888mmx;
#endif
				break;
			    case PICT_r5g6b5:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSrc_8888RevNPx0565mmx;
#endif
				break;
			    default:
				break;
			    }
			    break;
			default:
			    break;
			}
			break;
		    case PICT_x8r8g8b8:
			switch (pMask->format) {
			case PICT_a8r8g8b8:
			case PICT_a8b8g8r8:
			    switch (pDst->format) {
			    case PICT_a8b8g8r8:
			    case PICT_x8b8g8r8:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSrc_8888RevNPx8888mmx;
#endif
				break;
			    case PICT_r5g6b5:
#ifdef USE_MMX
				if (fbHaveMMX())
				    func = fbCompositeSrc_8888RevNPx0565mmx;
#endif
				break;
			    default:
				break;
			    }
			    break;
			default:
			    break;
			}
			break;
		    default:
			break;
		    }
		    break;
		}
		else if (maskRepeat &&
			 pMask->pDrawable->width == 1 &&
			 pMask->pDrawable->height == 1)
		{
		    switch (pSrc->format) {
		    case PICT_r5g6b5:
		    case PICT_b5g6r5:
			if (pDst->format == pSrc->format)
			    func = fbCompositeTrans_0565xnx0565;
			break;
		    case PICT_r8g8b8:
		    case PICT_b8g8r8:
			if (pDst->format == pSrc->format)
			    func = fbCompositeTrans_0888xnx0888;
			break;
#ifdef USE_MMX
		    case PICT_x8r8g8b8:
			if ((pDst->format == PICT_a8r8g8b8 ||
			     pDst->format == PICT_x8r8g8b8) &&
			    pMask->format == PICT_a8 && fbHaveMMX())
			    func = fbCompositeSrc_x888x8x8888mmx;
			break;
		    case PICT_x8b8g8r8:
			if ((pDst->format == PICT_a8b8g8r8 ||
			     pDst->format == PICT_x8b8g8r8) &&
			    pMask->format == PICT_a8 && fbHaveMMX())
			    func = fbCompositeSrc_x888x8x8888mmx;
			break;
		    case PICT_a8r8g8b8:
			if ((pDst->format == PICT_a8r8g8b8 ||
			     pDst->format == PICT_x8r8g8b8) &&
			    pMask->format == PICT_a8 && fbHaveMMX())
			    func = fbCompositeSrc_8888x8x8888mmx;
			break;
		    case PICT_a8b8g8r8:
			if ((pDst->format == PICT_a8b8g8r8 ||
			     pDst->format == PICT_x8b8g8r8) &&
			    pMask->format == PICT_a8 && fbHaveMMX())
			    func = fbCompositeSrc_8888x8x8888mmx;
			break;
#endif
		    default:
			break;
		    }
		    
		    if (func)
			maskRepeat = FALSE;
		}
	    }
	}
	else /* no mask */
	{
	    if (fbCanGetSolid(pSrc))
	    {
		/* no mask and repeating source */
		switch (pSrc->format) {
		case PICT_a8r8g8b8:
		    switch (pDst->format) {
		    case PICT_a8r8g8b8:
		    case PICT_x8r8g8b8:
#ifdef USE_MMX
			if (fbHaveMMX())
			{
			    srcRepeat = FALSE;
			    func = fbCompositeSolid_nx8888mmx;
			}
#endif
			break;
		    case PICT_r5g6b5:
#ifdef USE_MMX
			if (fbHaveMMX())
			{
			    srcRepeat = FALSE;
			    func = fbCompositeSolid_nx0565mmx;
			}
#endif
			break;
		    default:
			break;
		    }
		    break;
		default:
		    break;
		}
	    }
	    else if (! srcRepeat)
	    {
		/*
		 * Formats without alpha bits are just Copy with Over
		 */
		if (pSrc->format == pDst->format && !PICT_FORMAT_A(pSrc->format))
		{
#ifdef USE_MMX
		    if (fbHaveMMX() &&
			(pSrc->format == PICT_x8r8g8b8 || pSrc->format == PICT_x8b8g8r8))
			func = fbCompositeCopyAreammx;
		    else
#endif
			func = fbCompositeSrcSrc_nxn;
		}
		else switch (pSrc->format) {
		case PICT_a8r8g8b8:
		    switch (pDst->format) {
		    case PICT_a8r8g8b8:
		    case PICT_x8r8g8b8:
#ifdef USE_MMX
			if (fbHaveMMX())
			    func = fbCompositeSrc_8888x8888mmx;
			else
#endif
			    func = fbCompositeSrc_8888x8888;
			break;
		    case PICT_r8g8b8:
			func = fbCompositeSrc_8888x0888;
			break;
		    case PICT_r5g6b5:
#ifdef USE_MMX
			if (fbHaveMMX())
			    func = fbCompositeSrc_8888x0565mmx;
			else
#endif
			    func = fbCompositeSrc_8888x0565;
			break;
		    default:
			break;
		    }
		    break;
		case PICT_x8r8g8b8:
		    switch (pDst->format) {
		    case PICT_a8r8g8b8:
		    case PICT_x8r8g8b8:
#ifdef USE_MMX
			if (fbHaveMMX())
			    func = fbCompositeCopyAreammx;
#endif
			break;
		    default:
			break;
		    }
		case PICT_x8b8g8r8:
		    switch (pDst->format) {
		    case PICT_a8b8g8r8:
		    case PICT_x8b8g8r8:
#ifdef USE_MMX
			if (fbHaveMMX())
			    func = fbCompositeCopyAreammx;
#endif
			break;
		    default:
			break;
		    }
		    break;
		case PICT_a8b8g8r8:
		    switch (pDst->format) {
		    case PICT_a8b8g8r8:
		    case PICT_x8b8g8r8:
#ifdef USE_MMX
			if (fbHaveMMX())
			    func = fbCompositeSrc_8888x8888mmx;
			else
#endif
			    func = fbCompositeSrc_8888x8888;
			break;
		    case PICT_b8g8r8:
			func = fbCompositeSrc_8888x0888;
			break;
		    case PICT_b5g6r5:
#ifdef USE_MMX
			if (fbHaveMMX())
			    func = fbCompositeSrc_8888x0565mmx;
			else
#endif
			    func = fbCompositeSrc_8888x0565;
			break;
		    default:
			break;
		    }
		    break;
		default:
		    break;
		}
	    }
	}
	break;
    case PictOpAdd:
	if (pMask == 0)
	{
	    switch (pSrc->format) {
	    case PICT_a8r8g8b8:
		switch (pDst->format) {
		case PICT_a8r8g8b8:
#ifdef USE_MMX
		    if (fbHaveMMX())
			func = fbCompositeSrcAdd_8888x8888mmx;
		    else
#endif
			func = fbCompositeSrcAdd_8888x8888;
		    break;
		default:
		    break;
		}
		break;
	    case PICT_a8b8g8r8:
		switch (pDst->format) {
		case PICT_a8b8g8r8:
#ifdef USE_MMX
		    if (fbHaveMMX())
			func = fbCompositeSrcAdd_8888x8888mmx;
		    else
#endif
			func = fbCompositeSrcAdd_8888x8888;
		    break;
		default:
		    break;
		}
		break;
	    case PICT_a8:
		switch (pDst->format) {
		case PICT_a8:
#ifdef USE_MMX
		    if (fbHaveMMX())
			func = fbCompositeSrcAdd_8000x8000mmx;
		    else
#endif
			func = fbCompositeSrcAdd_8000x8000;
		    break;
		default:
		    break;
		}
		break;
	    case PICT_a1:
		switch (pDst->format) {
		case PICT_a1:
		    func = fbCompositeSrcAdd_1000x1000;
		    break;
		default:
		    break;
		}
		break;
	    default:
		break;
	    }
	}
	else
	{
	    if ((pSrc->format == PICT_a8r8g8b8	||
		 pSrc->format == PICT_a8b8g8r8) &&
		fbCanGetSolid (pSrc)		&&
		pMask->format == PICT_a8	&&
		pDst->format == PICT_a8)
	    {
		srcRepeat = FALSE;
#ifdef USE_MMX
		if (fbHaveMMX())
		    func = fbCompositeSrcAdd_8888x8x8mmx;
		else
#endif
		    func = fbCompositeSrcAdd_8888x8x8;
	    }
	}
	break;
    case PictOpSrc:
	if (pMask)
	{
#ifdef USE_MMX
	    if (fbCanGetSolid (pSrc))
	    {
		if (pMask->format == PICT_a8)
		{
		    switch (pDst->format)
		    {
		    case PICT_a8r8g8b8:
		    case PICT_x8r8g8b8:
		    case PICT_a8b8g8r8:
		    case PICT_x8b8g8r8:
			if (fbHaveMMX())
			{
			    srcRepeat = FALSE;
			    func = fbCompositeSolidMaskSrc_nx8x8888mmx;
			}
			break;
		    default:
			break;
		    }
		}
	    }
#endif
	}
	else
	{
	    if (pSrc->format == pDst->format)
	    {
#ifdef USE_MMX
		if (pSrc->pDrawable != pDst->pDrawable && fbHaveMMX() &&
		    (PICT_FORMAT_BPP (pSrc->format) == 16 ||
		     PICT_FORMAT_BPP (pSrc->format) == 32))
		    func = fbCompositeCopyAreammx;
		else
#endif
		    func = fbCompositeSrcSrc_nxn;
	    }
	}
	break;
    case PictOpIn:
#ifdef USE_MMX
	if (pSrc->format == PICT_a8 &&
	    pDst->format == PICT_a8 &&
	    !pMask)
	{
	    if (fbHaveMMX())
		func = fbCompositeIn_8x8mmx;
	}
	else if (srcRepeat && pMask && !pMask->componentAlpha &&
		 (pSrc->format == PICT_a8r8g8b8 ||
		  pSrc->format == PICT_a8b8g8r8)   &&
		 (pMask->format == PICT_a8)        &&
		 pDst->format == PICT_a8)
	{
	    if (fbHaveMMX())
	    {
		srcRepeat = FALSE;
		func = fbCompositeIn_nx8x8mmx;
	    }
	}
#else
	func = NULL;
#endif
       break;
    default:
	break;
    }

    if (!func) {
	func = fbCompositeRectWrapper;
    }

    /* if we are transforming, we handle repeats in fbFetchTransformed */
    if (srcTransform)
	srcRepeat = FALSE;
    if (maskTransform)
	maskRepeat = FALSE;
    
    if (!miComputeCompositeRegion (&region, pSrc, pMask, pDst, xSrc, ySrc,
				   xMask, yMask, xDst, yDst, width, height))
        return;

    n = REGION_NUM_RECTS (&region);
    pbox = REGION_RECTS (&region);
    while (n--)
    {
	h = pbox->y2 - pbox->y1;
	y_src = pbox->y1 - yDst + ySrc;
	y_msk = pbox->y1 - yDst + yMask;
	y_dst = pbox->y1;
	while (h)
	{
	    h_this = h;
	    w = pbox->x2 - pbox->x1;
	    x_src = pbox->x1 - xDst + xSrc;
	    x_msk = pbox->x1 - xDst + xMask;
	    x_dst = pbox->x1;
	    if (maskRepeat)
	    {
		y_msk = mod (y_msk - pMask->pDrawable->y, pMask->pDrawable->height);
		if (h_this > pMask->pDrawable->height - y_msk)
		    h_this = pMask->pDrawable->height - y_msk;
		y_msk += pMask->pDrawable->y;
	    }
	    if (srcRepeat)
	    {
		y_src = mod (y_src - pSrc->pDrawable->y, pSrc->pDrawable->height);
		if (h_this > pSrc->pDrawable->height - y_src)
		    h_this = pSrc->pDrawable->height - y_src;
		y_src += pSrc->pDrawable->y;
	    }
	    while (w)
	    {
		w_this = w;
		if (maskRepeat)
		{
		    x_msk = mod (x_msk - pMask->pDrawable->x, pMask->pDrawable->width);
		    if (w_this > pMask->pDrawable->width - x_msk)
			w_this = pMask->pDrawable->width - x_msk;
		    x_msk += pMask->pDrawable->x;
		}
		if (srcRepeat)
		{
		    x_src = mod (x_src - pSrc->pDrawable->x, pSrc->pDrawable->width);
		    if (w_this > pSrc->pDrawable->width - x_src)
			w_this = pSrc->pDrawable->width - x_src;
		    x_src += pSrc->pDrawable->x;
		}
		(*func) (op, pSrc, pMask, pDst,
			 x_src, y_src, x_msk, y_msk, x_dst, y_dst,
			 w_this, h_this);
		w -= w_this;
		x_src += w_this;
		x_msk += w_this;
		x_dst += w_this;
	    }
	    h -= h_this;
	    y_src += h_this;
	    y_msk += h_this;
	    y_dst += h_this;
	}
	pbox++;
    }
    REGION_UNINIT (pDst->pDrawable->pScreen, &region);
}