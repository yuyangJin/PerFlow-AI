'''
@module trace visualizer
'''

import numpy as np
import drawsvg as draw
from ..workflow import FlowNode

from ..core import Event, FwdBwdEvent, EventType, Trace

ENABLE_BORDER = True
ENABLE_BATCH_ID = True
ENABLE_EDGE_BLUR = False
SCALE_FACTOR = 2
S = SCALE_FACTOR

# TIME_PER_UNIT = 300 // SCALE_FACTOR
TIME_PER_UNIT = 4000 // SCALE_FACTOR

GREYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114])

TAIL = 50
BORDER_SIZE = 1
SPAN_HEIGHT = SCALE_FACTOR * 10
FONT_SIZE = SCALE_FACTOR * 10
TITLE_WIDTH = SCALE_FACTOR * 60
CENTER_TITLE_HEIGHT = SPAN_HEIGHT * 6

'''
@class TraceVisiualizer
A trace visualizer.
'''
class TraceVisiualizer(FlowNode):
    def __init__(self, trace):
        self.trace = trace

    def visualize(self):
        # visualize the trace
        #    Title Width              Data_width
        #   ------------------------------------------------
        #   |          |                                   |
        #   |   device |                                   |
        #   |   title  |              data                 |  data_height
        #   |          |                                   |
        #   |          |                                   |
        #   ------------------------------------------------
        #   |                    info                      |  info_height
        #   ------------------------------------------------
        #   |                  svg Title                   |  title_height
        #   ------------------------------------------------
        
        title_width = TITLE_WIDTH

        ndevs = 0
        devs = []
        exist_type = []
        first_time = 9223372036854775807
        last_time = 0
        max_chk = 1
        time_per_unit = TIME_PER_UNIT
        for it, dev_evs in enumerate(self.trace.m_events.values()):
            if len(dev_evs) != 0:
                ndevs = ndevs + 1
                devs.append(it)
                for e in dev_evs:
                    first_time = min(first_time, e.m_timestamp)
                    last_time = max(last_time, e.m_timestamp + e.m_duration)
                    max_chk = max(max_chk, e.m_chunk_id if hasattr(e, 'm_chunk_id') else 0)
                    time_per_unit = min(time_per_unit, (e.m_duration / (len(e.m_name) * FONT_SIZE * 0.3)))
                    if e.m_type not in exist_type:
                        exist_type.append(e.m_type)

        data_width = (last_time - first_time + time_per_unit - 1) // time_per_unit + TAIL
        
        data_height = SPAN_HEIGHT * ndevs + BORDER_SIZE * (ndevs + 1)
        if EventType.OFFL in exist_type:
            data_height = data_height * 3

        color_text_row_height = int(SPAN_HEIGHT * 1.6)
        color_text_height = int(SPAN_HEIGHT * 1.6) + BORDER_SIZE
        info_height = SPAN_HEIGHT + int(SPAN_HEIGHT * 1.6) + 3 * BORDER_SIZE

        title_height = CENTER_TITLE_HEIGHT

        canvas_width = title_width + data_width
        canvas_height = data_height + info_height + title_height

        d = draw.Drawing(canvas_width, canvas_height, origin="top-left")


        def get_color(typ, chk = 0):
            #COLOR_VALUE_MAP
            c = np.array([0, 0, 0])
            if typ == EventType.FWD:
                c = np.array([57, 122, 242])
            elif typ == EventType.BWD:
                c = np.array([62, 181, 191])
            elif typ == EventType.WGT:
                c = np.array([41, 137, 64])
            elif typ == EventType.COMM:
                c = np.array([255, 217, 102])
            elif typ == EventType.OPRT:
                c = np.array([255, 217, 102])
            elif typ == EventType.OFFL:
                c = np.array([227, 189, 152])
            elif typ == EventType.REL:
                c = np.array([198, 157, 225])
            else:
                raise ValueError("Type must be an instance of EventType")
            
            #lighten_color
            lightness = chk * 70 / max_chk
            
            R = c[0]
            G = c[1]
            B = c[2]  

            # Calculate the transformed RGB values 
            R_prime = int(R + (255 - R) * (lightness / 100))  
            G_prime = int(G + (255 - G) * (lightness / 100))  
            B_prime = int(B + (255 - B) * (lightness / 100))  

            # Ensure that the RGB values are within the range of 0 to 255 
            R_prime = max(0, min(255, R_prime))  
            G_prime = max(0, min(255, G_prime))  
            B_prime = max(0, min(255, B_prime))  

            return f"#{hex(R_prime)[2:]}{hex(G_prime)[2:]}{hex(B_prime)[2:]}"

        def plot_line(sy, sx, ey, ex, width=None):
            d.append(draw.Line(sx, sy, ex, ey, stroke='black', stroke_width=width or BORDER_SIZE))
        
        def plot_rect(sy, sx, h, w, color):
            d.append(draw.Rectangle(sx, sy, w, h, fill=color, shape_rendering="geometricPrecision"))

        def plot_rect_frame(sy, sx, h, w):
            d.append(draw.Rectangle(sx, sy, w, h, fill="none", stroke='black', stroke_width=BORDER_SIZE))
        
        def plot_text(y, x, text, anchor="middle", font_scale=1, fill='black'):
            font_size = FONT_SIZE * font_scale
            tl = len(text) * font_size // 2
            d.append(draw.Text(
                text, font_size,
                x,
                # Magic 3 to make it vertical center
                y + font_size - 3,
                textLength=tl, lengthAdjust='spacing',
                text_anchor=anchor,
                font_family="Times New Roman",
                fill=fill,
                # font_style="oblique",
                # font_family="Computer Modern Roman",
            ))

        def plot_arrow(start_y, start_x, width, thickness=2):
            b = thickness * (SCALE_FACTOR // 2)
            plot_line(start_y, start_x, start_y, start_x + width, b)
            plot_line(start_y, start_x + width, start_y - 3*b, start_x + width - 3*b)
            plot_line(start_y, start_x + width, start_y + 3*b, start_x + width - 3*b)

        def draw_device_title(oy, ox):
            for it, i in enumerate(devs):
                h = it * SPAN_HEIGHT + (it + 1) * BORDER_SIZE
                if EventType.OFFL in exist_type:
                    h = h * 3
                plot_text(oy + h, ox + 6 * SCALE_FACTOR, "Device {}".format(i), "left")
        
        def draw_data(oy, ox):
            for it, i in enumerate(devs):
                h = it * SPAN_HEIGHT + (it + 1) * BORDER_SIZE
                lh = SPAN_HEIGHT + BORDER_SIZE
                if EventType.OFFL in exist_type:
                    h = h * 3
                    lh = lh * 3
                
                dev_events = self.trace.get_events(i)
                for e in dev_events:
                    start = BORDER_SIZE + (e.m_timestamp - first_time) // time_per_unit
                    duration = e.m_duration // time_per_unit
                    center = start + duration // 2

                    bh = 0
                    if e.m_type == EventType.OFFL:
                        bh = SPAN_HEIGHT + BORDER_SIZE
                    elif e.m_type == EventType.REL:
                        bh = (SPAN_HEIGHT + BORDER_SIZE)*2
                    plot_rect(oy + h + bh, ox + start, SPAN_HEIGHT, duration, get_color(e.m_type, (e.m_chunk_id if hasattr(e, 'm_chunk_id') else 0)))
                    plot_rect_frame(oy + h + bh - BORDER_SIZE, ox + start, SPAN_HEIGHT + BORDER_SIZE, duration)
                    plot_text(oy + h + bh + SPAN_HEIGHT / 5, ox + center, e.m_name, font_scale=0.6, fill='black')
                plot_line(oy + h + lh, ox, oy + h + lh, ox + data_width -1)

        def draw_info(oy, ox, types):
            div = 2 + len(types)
            for it, typ in enumerate(types):
                start = canvas_width // div * (it + 1) 
                img_block_w = 25 * SCALE_FACTOR
                text_block_w = 28 * SCALE_FACTOR
                plot_rect(oy + color_text_height + BORDER_SIZE, ox + start, SPAN_HEIGHT, img_block_w, get_color(typ))
                plot_rect_frame(oy + color_text_height, ox + start, SPAN_HEIGHT + BORDER_SIZE, img_block_w)
                plot_text(oy + color_text_height, ox + start + text_block_w, str(f"{typ}".split('.')[1][0:3]).lower(), "left")

            plot_text(oy, 6 * SCALE_FACTOR, "Time", "left")
            plot_arrow(oy + SPAN_HEIGHT // 2 + BORDER_SIZE + 1, 65 * SCALE_FACTOR, 50 * SCALE_FACTOR)
            
        def draw_svg_title(oy, ox):
            plot_text(data_height + info_height + CENTER_TITLE_HEIGHT / 4, canvas_width / 2, "Title", "middle", 2)

        draw_device_title(0, 0)
        draw_data(0, title_width)
        draw_info(data_height, 0, exist_type)
        draw_svg_title(data_height + info_height, 0)

        d.save_svg("trace.svg")

    def run(self):
        self.visualize()
        pass
