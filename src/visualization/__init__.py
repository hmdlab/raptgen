from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib 
from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)

def provide_ax(func):
    from src.visualization import get_ax
    def wrapper_provide_ax(*args, **kwargs):
        no_ax_in_args = all(not isinstance(arg, matplotlib.axes.Axes) for arg in args)
        if no_ax_in_args and "ax" not in kwargs.keys():
            logger.info("ax not provided")
            fig, ax = get_ax(return_fig = True)
            kwargs["ax"] = ax
            kwargs["fig"] = fig
        func(*args, **kwargs)
    return wrapper_provide_ax


def get_results_df(result_dir:str) -> pd.DataFrame:
    """get results in the dir and make to dataframe with specified naming rule"""
    result = list()
    result_dir = Path(result_dir)
    for filepath in result_dir.glob("*.csv"):
        df = pd.read_csv(filepath)
        z = int(filepath.stem.split("_")[2].replace("z",""))
        epoch,tr_loss,te_loss,p,kld =  df.iloc[np.argmin(df.test_loss)]
        result.append((z,te_loss,filepath))
    result_df = pd.DataFrame(result,columns=["z","min_loss","path"])
    return result_df

def get_ax(row_col=(1,1),dpi= 150, figsize=(10,6), return_fig=False):
    fig, ax = plt.subplots(*row_col, dpi= dpi, figsize=figsize)
    if return_fig:
        return fig, ax
    else :
        return ax

def plot_violinplot(data, ax = None):
    """given N x 2 data with index first and score second, plot a violinplot"""
    assert data.shape[1] == 2, "is this [:,2] shaped numpy array?"
    
    if ax is None:
        logger.info("no ax specified")
        fig, ax = plt.subplots(1,1,dpi=150, figsize=(10,6))
    dims = np.unique(data[:,0])
    
    dataset = [data[data[:,0] == i][:,1] for i in dims]
    
    ax.scatter(*data.T, s=5,c="k")
    ax.violinplot (dataset)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Embedding dim of Variational Autoencoder")
    ax.set_title(f"Model Loss change of z dimension (N={sum(data[:,0] == dims[0])})")
    
    return ax


def write_profile_hmm_svg(a, e_m,
                          name="profile_HMM.svg", dx=140,
                          width=70,
                          dxdyratio=0.6, savepng=True, is_log_proba=True, stroke_w = 20):
    from cairosvg import svg2png
    import svgwrite
    from src.data import State, Transition
    if is_log_proba:
        a = np.exp(a)
        e_m = np.exp(e_m)

    MOTIF_LEN = a.shape[0] - 1

    GREEN = (92, 209, 62)
    RED = (195, 41, 28)
    YELLOW = (244, 173, 61)
    BLUE = (0, 18, 184)
    COLOR = np.array([GREEN, RED, YELLOW, BLUE])

    nt_color = (e_m @ COLOR).astype('uint8')


    y_insertion = dx * dxdyratio + width//8
    y_deletion  = dx * dxdyratio * 2
    y_match     = dx * dxdyratio * 3

    dwg = svgwrite.Drawing(filename=name, debug=True,
                           size=(dx*(MOTIF_LEN+4), y_match+width+140))

    lines = dwg.add(dwg.g(id='lines', fill='white', stroke="black"))
    shapes = dwg.add(dwg.g(id='shapes', fill='white',
                           stroke="black", stroke_width=5))
    texts = dwg.add(dwg.g(id='texts'))
    centers = []

    for i in range(MOTIF_LEN+2):
        x_start = dx*(i+1)
        x_end = x_start + width
        x_center = (x_start+x_end)//2

        x_insertion_center = x_center + dx//2

        centers.append([(x_center, y_match+width//2),
                        (x_insertion_center, y_insertion), (x_center, y_deletion)])


        # Match to Match
        if i != 0:
            lines.add(dwg.line(centers[i-1][State.M], centers[i]
                               [State.M], stroke_width=stroke_w*a[i-1, Transition.M2M]))
        # Match to Insertion
        if i != MOTIF_LEN+1:
            lines.add(dwg.line(centers[i][State.M], centers[i]
                               [State.I], stroke_width=stroke_w*a[i, Transition.M2I]))
        # Match to Deletion
        if i != MOTIF_LEN+1 and i != 0:
            lines.add(dwg.line(centers[i-1][State.M], centers[i]
                               [State.D], stroke_width=stroke_w*a[i-1, Transition.M2D]))
        # Insertion to Match
        if i != 0:
            lines.add(dwg.line(centers[i-1][State.I], centers[i]
                               [State.M], stroke_width=stroke_w*a[i-1, Transition.I2M]))

        # Insertion to Insertion
        if i != MOTIF_LEN+1:
            lines.add(dwg.ellipse(center=(x_center-width//2+dx//2, y_insertion), r=(width //
                                                                          4, width//4), stroke_width=stroke_w*a[i, Transition.I2I]))
        # Deletion to Match
        if i > 1:
            lines.add(dwg.line(centers[i-1][State.D], centers[i]
                               [State.M], stroke_width=stroke_w*a[i-1, Transition.D2M]))

        
        # Deletion to Deletion
        if i > 1 and i != MOTIF_LEN+1:
            lines.add(dwg.line(centers[i-1][State.D], centers[i]
                               [State.D], stroke_width=stroke_w*a[i-1, Transition.D2D]))

        # match
        if MOTIF_LEN+1 > i > 0:
            r, g, b = nt_color[i-1]
        else:
            r, g, b = 255, 255, 255
        shapes.add(dwg.polygon([(x_start, y_match), (x_end, y_match), (x_end, y_match+width), (x_start, y_match+width)],
                               fill=f"rgb({r},{g},{b})"))

        # insertion
        if i != MOTIF_LEN+1:
            shapes.add(dwg.polygon([(x_start+dx//2, y_insertion), (x_center+dx//2, y_insertion -
                                                         width//2), (x_end+dx//2, y_insertion), (x_center+dx//2, y_insertion+width//2)]))

        # deletion
        if i != 0 and i != MOTIF_LEN+1:
            shapes.add(dwg.ellipse(
                center=(x_center, y_deletion), r=(width//2, width//2)))

        texts.add(dwg.text("A", (x_start, y_match+width+40)))
        texts.add(dwg.text("T", (x_start, y_match+width+60)))
        texts.add(dwg.text("G", (x_start, y_match+width+80)))
        texts.add(dwg.text("C", (x_start, y_match+width+100)))

        if 0 < i < MOTIF_LEN+1:
            for j in range(4):
                lines.add(dwg.line(
                    (x_start + width//4, y_match+width+34+20*j),
                    (x_start + width//4+(width*3/4) *
                     e_m[i-1, j], y_match+width+34+20*j),
                    stroke_width=12))

    dwg.save()
    logger.info(f"saved to {name}")
    if savepng:
        with open(name) as f:
            name_png = name.replace(".svg", ".png")
            svg2png(f.read(), write_to=name_png)
            logger.info(f"saved to {name_png}")



class SeqLogoDrawer():
    """based on https://weblogo.berkeley.edu/
    """

    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
    
    def __init__(self, isRNA=True):
        red    = "#d50000"
        green  = "#00d500"
        blue   = "#0000c0"
        yellow = "#ffaa00"
        res = dict()
        if isRNA:
            self.string = list("AUGC")
        else:
            self.string = list("ATGC")

        for text,color in zip(self.string,[green,red,yellow,blue]):
            img = Image.new('RGB', (3000, 1500),(255,255,255))
            d = ImageDraw.Draw(img)
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",size=800)
            d.text((10, 10), text, fill=color,font=font)

            img_sum = np.sum(255-np.asarray(img),axis=2)

            sw = False
            for idx,i in enumerate(np.sum(img_sum,axis=1)):
                if i != 0 and sw is False:
                    sw = True
                    x_start = idx
                if i ==0 and sw is True:
                    sw = False
                    x_end = idx

            sw = False
            for idx,i in enumerate(np.sum(img_sum,axis=0)):
                if i != 0 and sw is False:
                    sw = True
                    y_start = idx
                if i == 0 and sw is True:
                    sw = False
                    y_end = idx

            #画像をarrayに変換
            im_list = np.asarray(img)[x_start:x_end,y_start:y_end]
            res[text]=im_list

        self.res = res
   
    def draw_logo(self, seq, ax=None, calc_h_em=True, correction = 0):
        import numpy as np
        assert seq.shape[0] == 4, "is this size==(4,length) numpy array?"
        assert np.all(correction == 0) or seq.shape[1] == len(correction), "the correction should be same with sequence length"
        _, length = seq.shape
       
        if calc_h_em:
            p = seq
            H = -p * np.log2(p+1e-30)
            R = 2 - np.sum(H,axis=0, keepdims=True) - correction
            h_em = p * R
        else:
            h_em = seq 
        # logger.info(h_em)
        width = 200
        c_h = width * 5 * 2
        c_w = width * h_em.shape[1]

        canvas = np.ones((c_h,c_w,3))
       
        w_offset = 0
        for idx,(a,t,g,c) in enumerate(h_em.T):
            # logger.info(idx,a,t,g,c)
            h_offset = 0
            for i in np.argsort([a,t,g,c])[::-1]:
                w,h = width, int(width*5*[a,t,g,c][i])
                if h != 0:
                    canvas[c_h - h_offset - h : c_h-h_offset,w_offset:w_offset+w] = np.asarray(Image.fromarray(self.res[self.string[i]]).resize((w,h), Image.LANCZOS))/255
                h_offset += h
            w_offset += w
            # break
        if ax is None:
            plt.figure(dpi=150,figsize=(20,6))
            plt.xticks(np.arange(length)*width+width//2,1+np.arange(length))
            plt.yticks(np.arange(5)*width*5//2,2-np.arange(5)/2)
            plt.grid(False)
            plt.imshow(canvas)
        else:
            ax.imshow(canvas)
            ax.set_xticks(np.arange(length)*width+width//2)
            ax.set_xticklabels(1+np.arange(length))
            ax.set_yticks(np.arange(5)*width*5//2)
            ax.set_yticklabels(2-np.arange(5)/2)
            ax.grid(False)

@provide_ax
def draw_most_probable(a, e_m, ax, fig = None, header="phmm", save=True, force = False, drawer=None, draw_only_insertion=True):

    from src.data import State, Transition
    from src.data import ProfileHMMSampler
    if drawer is None:
        drawer = SeqLogoDrawer()

    sampler = ProfileHMMSampler(a, e_m, proba_is_log=True)
    proba = []
    xlabels = []
    for i, state in sampler.most_probable()[0]:
        # logger.info(i,state, state==State.M,state==State.I)
        if 0 < i <= e_m.shape[0]:
            if state == State.M:
                proba.append(np.exp(e_m[i-1]))
                if draw_only_insertion:
                    xlabels += [" "]
                else:
                    xlabels += ["M"]
            elif state == State.I:
                proba.append(np.ones((4))*0.25)
                if draw_only_insertion:
                    xlabels += ["^"]
                else:
                    xlabels += ["I"]
    drawer.draw_logo(np.stack(proba).T, ax=ax)
    ax.set_xticklabels(xlabels)
    ax.set_title(f"Most probable sequence derived from most probable states")
    if save and not force and Path(f"{header}.png").exists():
        logger.info(f"you are about to override {header}.png.  use force=True to override output file.")
        raise AssertionError
    if save:
        plt.savefig(f"{header}.png", bbox_inches='tight')
    return ax 