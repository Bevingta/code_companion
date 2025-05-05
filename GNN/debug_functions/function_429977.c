term_enter_normal_mode(void)
{
    term_T *term = curbuf->b_term;

    set_terminal_mode(term, TRUE);

    /* Append the current terminal contents to the buffer. */
    may_move_terminal_to_buffer(term, TRUE);

    /* Move the window cursor to the position of the cursor in the
     * terminal. */
    curwin->w_cursor.lnum = term->tl_scrollback_scrolled
					     + term->tl_cursor_pos.row + 1;
    check_cursor();
    if (coladvance(term->tl_cursor_pos.col) == FAIL)
	coladvance(MAXCOL);

    /* Display the same lines as in the terminal. */
    curwin->w_topline = term->tl_scrollback_scrolled + 1;
}