hfs_cat_traverse(HFS_INFO * hfs,
    TSK_HFS_BTREE_CB a_cb, void *ptr)
{
    TSK_FS_INFO *fs = &(hfs->fs_info);
    uint32_t cur_node;          /* node id of the current node */
    char *node;

    uint16_t nodesize;
    uint8_t is_done = 0;

    tsk_error_reset();

    nodesize = tsk_getu16(fs->endian, hfs->catalog_header.nodesize);
    if ((node = (char *) tsk_malloc(nodesize)) == NULL)
        return 1;

    /* start at root node */
    cur_node = tsk_getu32(fs->endian, hfs->catalog_header.rootNode);

    /* if the root node is zero, then the extents btree is empty */
    /* if no files have overflow extents, the Extents B-tree still
       exists on disk, but is an empty B-tree containing only
       the header node */
    if (cur_node == 0) {
        if (tsk_verbose)
            tsk_fprintf(stderr, "hfs_cat_traverse: "
                "empty extents btree\n");
        free(node);
        return 1;
    }

    if (tsk_verbose)
        tsk_fprintf(stderr, "hfs_cat_traverse: starting at "
            "root node %" PRIu32 "; nodesize = %"
            PRIu16 "\n", cur_node, nodesize);

    /* Recurse down to the needed leaf nodes and then go forward */
    is_done = 0;
    while (is_done == 0) {
        TSK_OFF_T cur_off;      /* start address of cur_node */
        uint16_t num_rec;       /* number of records in this node */
        ssize_t cnt;
        hfs_btree_node *node_desc;

        // sanity check
        if (cur_node > tsk_getu32(fs->endian,
                hfs->catalog_header.totalNodes)) {
            tsk_error_set_errno(TSK_ERR_FS_GENFS);
            tsk_error_set_errstr
                ("hfs_cat_traverse: Node %d too large for file", cur_node);
            free(node);
            return 1;
        }

        // read the current node
        cur_off = cur_node * nodesize;
        cnt = tsk_fs_attr_read(hfs->catalog_attr, cur_off,
            node, nodesize, 0);
        if (cnt != nodesize) {
            if (cnt >= 0) {
                tsk_error_reset();
                tsk_error_set_errno(TSK_ERR_FS_READ);
            }
            tsk_error_set_errstr2
                ("hfs_cat_traverse: Error reading node %d at offset %"
                PRIuOFF, cur_node, cur_off);
            free(node);
            return 1;
        }

        // process the header / descriptor
        if (nodesize < sizeof(hfs_btree_node)) {
            tsk_error_set_errno(TSK_ERR_FS_GENFS);
            tsk_error_set_errstr
            ("hfs_cat_traverse: Node size %d is too small to be valid", nodesize);
            free(node);
            return 1;
        }
        node_desc = (hfs_btree_node *) node;
        num_rec = tsk_getu16(fs->endian, node_desc->num_rec);

        if (tsk_verbose)
            tsk_fprintf(stderr, "hfs_cat_traverse: node %" PRIu32
                " @ %" PRIu64 " has %" PRIu16 " records\n",
                cur_node, cur_off, num_rec);

        if (num_rec == 0) {
            tsk_error_set_errno(TSK_ERR_FS_GENFS);
            tsk_error_set_errstr("hfs_cat_traverse: zero records in node %"
                PRIu32, cur_node);
            free(node);
            return 1;
        }

        /* With an index node, find the record with the largest key that is smaller
         * to or equal to cnid */
        if (node_desc->type == HFS_BT_NODE_TYPE_IDX) {
            uint32_t next_node = 0;
            int rec;

            for (rec = 0; rec < num_rec; ++rec) {
                size_t rec_off;
                hfs_btree_key_cat *key;
                uint8_t retval;
                uint16_t keylen;

                // get the record offset in the node
                rec_off =
                    tsk_getu16(fs->endian,
                    &node[nodesize - (rec + 1) * 2]);
                if (rec_off > nodesize) {
                    tsk_error_set_errno(TSK_ERR_FS_GENFS);
                    tsk_error_set_errstr
                        ("hfs_cat_traverse: offset of record %d in index node %d too large (%d vs %"
                        PRIu16 ")", rec, cur_node, (int) rec_off,
                        nodesize);
                    free(node);
                    return 1;
                }

                key = (hfs_btree_key_cat *) & node[rec_off];

                keylen = 2 + tsk_getu16(hfs->fs_info.endian, key->key_len);
                if ((keylen) > nodesize) {
                    tsk_error_set_errno(TSK_ERR_FS_GENFS);
                    tsk_error_set_errstr
                        ("hfs_cat_traverse: length of key %d in index node %d too large (%d vs %"
                        PRIu16 ")", rec, cur_node, keylen, nodesize);
                    free(node);
                    return 1;
                }


                /*
                   if (tsk_verbose)
                   tsk_fprintf(stderr,
                   "hfs_cat_traverse: record %" PRIu16
                   " ; keylen %" PRIu16 " (%" PRIu32 ")\n", rec,
                   tsk_getu16(fs->endian, key->key_len),
                   tsk_getu32(fs->endian, key->parent_cnid));
                 */


                /* save the info from this record unless it is too big */
                retval =
                    a_cb(hfs, HFS_BT_NODE_TYPE_IDX, key,
                    cur_off + rec_off, ptr);
                if (retval == HFS_BTREE_CB_ERR) {
                    tsk_error_set_errno(TSK_ERR_FS_GENFS);
                    tsk_error_set_errstr2
                        ("hfs_cat_traverse: Callback returned error");
                    free(node);
                    return 1;
                }
                // record the closest entry
                else if ((retval == HFS_BTREE_CB_IDX_LT)
                    || (next_node == 0)) {
                    hfs_btree_index_record *idx_rec;
                    int keylen =
                        2 + hfs_get_idxkeylen(hfs, tsk_getu16(fs->endian,
                            key->key_len), &(hfs->catalog_header));
                    if (rec_off + keylen > nodesize) {
                        tsk_error_set_errno(TSK_ERR_FS_GENFS);
                        tsk_error_set_errstr
                            ("hfs_cat_traverse: offset of record and keylength %d in index node %d too large (%d vs %"
                            PRIu16 ")", rec, cur_node,
                            (int) rec_off + keylen, nodesize);
                        free(node);
                        return 1;
                    }
                    idx_rec =
                        (hfs_btree_index_record *) & node[rec_off +
                        keylen];
                    next_node = tsk_getu32(fs->endian, idx_rec->childNode);
                }
                if (retval == HFS_BTREE_CB_IDX_EQGT) {
                    // move down to the next node
                    break;
                }
            }
            // check if we found a relevant node
            if (next_node == 0) {
                tsk_error_set_errno(TSK_ERR_FS_GENFS);
                tsk_error_set_errstr
                    ("hfs_cat_traverse: did not find any keys in index node %d",
                    cur_node);
                is_done = 1;
                break;
            }
            // TODO: Handle multinode loops
            if (next_node == cur_node) {
                tsk_error_set_errno(TSK_ERR_FS_GENFS);
                tsk_error_set_errstr
                    ("hfs_cat_traverse: node %d references itself as next node",
                    cur_node);
                is_done = 1;
                break;
            }
            cur_node = next_node;
        }

        /* With a leaf, we look for the specific record. */
        else if (node_desc->type == HFS_BT_NODE_TYPE_LEAF) {
            int rec;

            for (rec = 0; rec < num_rec; ++rec) {
                size_t rec_off;
                hfs_btree_key_cat *key;
                uint8_t retval;
                uint16_t keylen;

                // get the record offset in the node
                rec_off =
                    tsk_getu16(fs->endian,
                    &node[nodesize - (rec + 1) * 2]);
                if (rec_off > nodesize) {
                    tsk_error_set_errno(TSK_ERR_FS_GENFS);
                    tsk_error_set_errstr
                        ("hfs_cat_traverse: offset of record %d in leaf node %d too large (%d vs %"
                        PRIu16 ")", rec, cur_node, (int) rec_off,
                        nodesize);
                    free(node);
                    return 1;
                }
                key = (hfs_btree_key_cat *) & node[rec_off];

                keylen = 2 + tsk_getu16(hfs->fs_info.endian, key->key_len);
                if ((keylen) > nodesize) {
                    tsk_error_set_errno(TSK_ERR_FS_GENFS);
                    tsk_error_set_errstr
                        ("hfs_cat_traverse: length of key %d in leaf node %d too large (%d vs %"
                        PRIu16 ")", rec, cur_node, keylen, nodesize);
                    free(node);
                    return 1;
                }

                /*
                   if (tsk_verbose)
                   tsk_fprintf(stderr,
                   "hfs_cat_traverse: record %" PRIu16
                   "; keylen %" PRIu16 " (%" PRIu32 ")\n", rec,
                   tsk_getu16(fs->endian, key->key_len),
                   tsk_getu32(fs->endian, key->parent_cnid));
                 */
                //                rec_cnid = tsk_getu32(fs->endian, key->file_id);

                retval =
                    a_cb(hfs, HFS_BT_NODE_TYPE_LEAF, key,
                    cur_off + rec_off, ptr);
                if (retval == HFS_BTREE_CB_LEAF_STOP) {
                    is_done = 1;
                    break;
                }
                else if (retval == HFS_BTREE_CB_ERR) {
                    tsk_error_set_errno(TSK_ERR_FS_GENFS);
                    tsk_error_set_errstr2
                        ("hfs_cat_traverse: Callback returned error");
                    free(node);
                    return 1;
                }
            }

            // move right to the next node if we got this far
            if (is_done == 0) {
                cur_node = tsk_getu32(fs->endian, node_desc->flink);
                if (cur_node == 0) {
                    is_done = 1;
                }
                if (tsk_verbose)
                    tsk_fprintf(stderr,
                        "hfs_cat_traverse: moving forward to next leaf");
            }
        }
        else {
            tsk_error_set_errno(TSK_ERR_FS_GENFS);
            tsk_error_set_errstr("hfs_cat_traverse: btree node %" PRIu32
                " (%" PRIu64 ") is neither index nor leaf (%" PRIu8 ")",
                cur_node, cur_off, node_desc->type);
            free(node);
            return 1;
        }
    }
    free(node);
    return 0;
}