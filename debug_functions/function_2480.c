static int rtmp_packet_read_one_chunk(URLContext *h, RTMPPacket *p,
                                      int chunk_size, RTMPPacket **prev_pkt_ptr,
                                      int *nb_prev_pkt, uint8_t hdr)
{

    uint8_t buf[16];
    int channel_id, timestamp, size;
    uint32_t ts_field; // non-extended timestamp or delta field
    uint32_t extra = 0;
    enum RTMPPacketType type;
    int written = 0;
    int ret, toread;
    RTMPPacket *prev_pkt;

    written++;
    channel_id = hdr & 0x3F;

    if (channel_id < 2) { //special case for channel number >= 64
        buf[1] = 0;
        if (ffurl_read_complete(h, buf, channel_id + 1) != channel_id + 1)
            return AVERROR(EIO);
        written += channel_id + 1;
        channel_id = AV_RL16(buf) + 64;
    }
    if ((ret = ff_rtmp_check_alloc_array(prev_pkt_ptr, nb_prev_pkt,
                                         channel_id)) < 0)
        return ret;
    prev_pkt = *prev_pkt_ptr;
    size  = prev_pkt[channel_id].size;
    type  = prev_pkt[channel_id].type;
    extra = prev_pkt[channel_id].extra;

    hdr >>= 6; // header size indicator
    if (hdr == RTMP_PS_ONEBYTE) {
        ts_field = prev_pkt[channel_id].ts_field;
    } else {
        if (ffurl_read_complete(h, buf, 3) != 3)
            return AVERROR(EIO);
        written += 3;
        ts_field = AV_RB24(buf);
        if (hdr != RTMP_PS_FOURBYTES) {
            if (ffurl_read_complete(h, buf, 3) != 3)
                return AVERROR(EIO);
            written += 3;
            size = AV_RB24(buf);
            if (ffurl_read_complete(h, buf, 1) != 1)
                return AVERROR(EIO);
            written++;
            type = buf[0];
            if (hdr == RTMP_PS_TWELVEBYTES) {
                if (ffurl_read_complete(h, buf, 4) != 4)
                    return AVERROR(EIO);
                written += 4;
                extra = AV_RL32(buf);
            }
        }
    }
    if (ts_field == 0xFFFFFF) {
        if (ffurl_read_complete(h, buf, 4) != 4)
            return AVERROR(EIO);
        timestamp = AV_RB32(buf);
    } else {
        timestamp = ts_field;
    }
    if (hdr != RTMP_PS_TWELVEBYTES)
        timestamp += prev_pkt[channel_id].timestamp;

    if (!prev_pkt[channel_id].read) {
        if ((ret = ff_rtmp_packet_create(p, channel_id, type, timestamp,
                                         size)) < 0)
            return ret;
        p->read = written;
        p->offset = 0;
        prev_pkt[channel_id].ts_field   = ts_field;
        prev_pkt[channel_id].timestamp  = timestamp;
    } else {
        // previous packet in this channel hasn't completed reading
        RTMPPacket *prev = &prev_pkt[channel_id];
        p->data          = prev->data;
        p->size          = prev->size;
        p->channel_id    = prev->channel_id;
        p->type          = prev->type;
        p->ts_field      = prev->ts_field;
        p->extra         = prev->extra;
        p->offset        = prev->offset;
        p->read          = prev->read + written;
        p->timestamp     = prev->timestamp;
        prev->data       = NULL;
    }
    p->extra = extra;
    // save history
    prev_pkt[channel_id].channel_id = channel_id;
    prev_pkt[channel_id].type       = type;
    prev_pkt[channel_id].size       = size;
    prev_pkt[channel_id].extra      = extra;
    size = size - p->offset;

    toread = FFMIN(size, chunk_size);
    if (ffurl_read_complete(h, p->data + p->offset, toread) != toread) {
        ff_rtmp_packet_destroy(p);
        return AVERROR(EIO);
    }
    size      -= toread;
    p->read   += toread;
    p->offset += toread;

    if (size > 0) {
       RTMPPacket *prev = &prev_pkt[channel_id];
       prev->data = p->data;
       prev->read = p->read;
       prev->offset = p->offset;
       p->data      = NULL;
       return AVERROR(EAGAIN);
    }

    prev_pkt[channel_id].read = 0; // read complete; reset if needed
    return p->read;
}