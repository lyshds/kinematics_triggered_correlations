import flylib as flb
#from thllib import flylib as flb
import flylib as flb
from matplotlib import pyplot as plt
import numpy as np
import scipy
from flylib import util
import figurefirst as fifi
import scipy.signal
#import local_project_functions as lpf
from IPython.display import SVG,display

import networkx as nx

flynumbers = list(range(1389,1402))
#flynumbers = list(range(1548,1549))
flylist = [flb.NetFly(fnum,rootpath='/media/imager/FlyDataD/FlyDB/') for fnum in flynumbers]
l = [fly.open_signals() for fly in flylist]
#fly = flylist[4]

#fly = flylist[0]
for fly in range(len(flylist)):
    print("%s_%s" % (fly, 'flydf'))
    name=("%s_%s" % (fly,'flydf')) 
    str(value(name))=fly.construct_dataframe()
    #flydf_=fly.construct_dataframe()
    #flydf_=fly.construct_dataframe()
#Access calcium values for a specific muscle and specific stimulus
pretrial_stripe_fix_b2_right = flydf.loc[
    flydf['stimulus']=='pretrial_stripe_fix',['b2_right']]

print(np.shape(pretrial_stripe_fix_b2_right))

#https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
#print(filtered_df[key+'_right'])

general_sorted_keys = sorted(fly.ca_cam_left_model_fits.keys())
print(sorted(fly.ca_cam_left_model_fits.keys()))

sorted_keys = []

for key in general_sorted_keys:
    key2= key+'_right'
    key3= key+'_left'
    sorted_keys.append(key2)
    sorted_keys.append(key3)
    
print(sorted_keys)

filtered_df = flydf.loc[flydf['stimulus']=='cl_blocks, g_x=-1, g_y=0, b_x=-8, b_y=0, ch=True']
#filtered_df= flydf

print(filtered_df.head())

layout = fifi.FigureLayout('graph_layout.svg',make_mplfigures=True)


cull_list = [('left', 'bkg'),('right', 'bkg'),
            ('left', 'iii24'),('right', 'iii24'),
            ('left', 'nm'),('right', 'nm'),
            ('left', 'pr'),('right', 'pr'),
            ('left', 'tpd'),('right', 'tpd')]

for cull in cull_list:
    sorted_keys.remove(cull[1]+'_'+cull[0])
#[sorted_keys.remove(cull) for cull in cull_list]

graphs = {}
for fly in flylist:
    state_mtrx = np.vstack([filtered_df[key] for key in sorted_keys])
    centered_mtrx = state_mtrx - np.mean(state_mtrx,axis = 1)[:,None]
    std_mtrx = centered_mtrx/np.std(centered_mtrx,axis = 1)[:,None]
    cor_mtrx = np.dot(std_mtrx,std_mtrx.T)
    G = nx.Graph()
    for i,lbl1 in enumerate(sorted_keys):
        for j,lbl2 in enumerate(sorted_keys):
            G.add_edge(lbl1,lbl2,weight = cor_mtrx[i,j])
    graphs[fly.flynum] = G

edges = G.edges
c_ex = layout.pathspecs['excitatory'].mplkwargs()['edgecolor']
c_in = layout.pathspecs['inhibitory'].mplkwargs()['edgecolor']
colors = [{True:c_ex,False:c_in}[G[e[0]][e[1]]['weight']>0.] for e in edges]


h = float(layout.layout_uh)
pos_dict = {}
for n in G.nodes:
    #n_s = '%s_%s'%(n[0][0].capitalize(),n[1])
    n1, n2 = n.split('_')
    n_s = '%s_%s'%(n2[0].capitalize(), n1)
    cx = float(layout.pathspecs[n_s]['cx'])
    cy = h-float(layout.pathspecs[n_s]['cy'])
    try:
        if 'transform' in layout.pathspecs[n_s].keys():
            t1 = fifi.svg_to_axes.parse_transform(layout.pathspecs[n_s]['transform'])
            p = np.dot(t1,np.array([cx,cy,1]))
            pos_dict[n] = (p[0],p[1])
        else:
            pos_dict[n]  = (cx,cy)
    except KeyError:
        print n

for flynum,G in graphs.items():
    edges= G.edges
    weights = [np.abs(G[e[0]][e[1]]['weight'])**2.6/100000000000. for e in edges]
    nx.draw(G,
            ax = layout.axes['network_graph_layout'],
            pos = pos_dict,
            font_color = 'r',
            with_labels= False,
            width = weights,
            edge_color = colors,
            node_color = 'k',
            alpha = 0.1)

    
    
layout.axes['network_graph_layout'].set_ybound(0,layout.axes['network_graph_layout'].h)
layout.axes['network_graph_layout'].set_xbound(0,layout.axes['network_graph_layout'].w)

layout.save('graph.svg')
plt.close('all')
