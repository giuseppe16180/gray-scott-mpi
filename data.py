# %%
import os

import pandas as pd
import plotly
import plotly.graph_objects as go
from functional import seq as s
from plotly import express as px
from plotly.subplots import make_subplots

files = [f for f in os.listdir('times') if not f.startswith('.')]

resolutions = s(files).sorted().group_by(lambda x: x[0:3]).sorted().to_dict()


def parse(file):
    return (
        s.open(f'times/{file}')
        .map(lambda x: x.strip()[1:-1].split(','))
        .map(lambda x: (x[0][1:-1], float(x[1])))
        .group_by_key()
        .map(lambda x: (x[0], sum(x[1])))
    ).to_list()


dataframe = []

for edge_size, files in resolutions.items():
    for file in files:
        x = parse(file)
        x.append(('edge_size', edge_size))
        x.append(('cores', file.split('.')[0][-1]))
        sorted(x)
        dataframe.append(x)

# %%
header = ['send recv time', 'gather time', 'total time', 'edge size', 'cores']
df = s(dataframe).map(lambda x: s(x).map(lambda x: x[1])).to_pandas(header)

# %%
# tempo impiegato in funzione del numero di core
fig = px.bar(df, x='edge size', y='total time', color='cores', barmode='group',
             labels={"cores": "num. core ", "edge size": "dimensione lato", "total time": "tempo totale"})


fig.update_layout(
    font=dict(size=18))

fig.show()
fig.write_image(f'plots/tempi.svg')
plotly.offline.plot(fig, filename='plots/tempi.html')


# %% speedup
sdf = df[['total time', 'edge size', 'cores']]

sdf = sdf.merge(sdf, on='edge size')
sdf = sdf[sdf['cores_x'] == '0']
sdf = sdf[sdf['cores_y'] != '0'].reset_index(drop=True)
sdf['speedup'] = sdf['total time_x'] / sdf['total time_y']
sdf['efficiency'] = sdf['speedup'] / sdf['cores_y'].astype(int)
sdf = sdf.rename(columns={'cores_y': 'n. cores'})
sdf.reset_index()

fig = px.bar(sdf, x='edge size', y='speedup', color='n. cores', barmode='group',
             labels={"n. cores": "num. core", "edge size": "dimensione lato", "total time": "tempo totale"})


fig.update_layout(
    font=dict(size=18))
fig.write_image(f'plots/speedup.svg')
plotly.offline.plot(fig, filename='plots/speedup.html')

# %%
fig = px.bar(sdf, x='n. cores', y='efficiency', color='edge size', barmode='group',
             labels={"n. cores": "num. core", "edge size": "dimensione lato", "efficiency": "efficienza"})


fig.update_layout(
    font=dict(size=18))

fig.show()
fig.write_image(f'plots/efficienza.svg')
plotly.offline.plot(fig, filename='plots/efficienza.html')


# %%
# percentage of the time spent in the send_recv

pdf = df.copy()

# .apply(lambda x: round(x * 100, 2))
pdf['send recv'] = (pdf['send recv time'])
pdf['gather'] = (pdf['gather time'])  # .apply(lambda x: round(x * 100, 2))
pdf['altro'] = (pdf['total time'] - pdf['send recv'] - pdf['gather'])
pdf = pdf[['edge size', 'cores', 'send recv', 'gather', 'altro']]

# transpose a column to several rows
pdf = pdf.melt(id_vars=['edge size', 'cores'],
               var_name='operazione', value_name='tempo')


def get_tempi(lato):

    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'domain'}, {
                        'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]])

    temp = pdf[(pdf['edge size'] == lato) & (pdf['cores'] == '1')]

    fig.add_trace(go.Pie(labels=temp['operazione'], values=temp['tempo'], title='1 core',
                         marker_colors=temp['operazione']), 1, 1)

    temp = pdf[(pdf['edge size'] == lato) & (pdf['cores'] == '2')]

    fig.add_trace(go.Pie(labels=temp['operazione'], values=temp['tempo'], title='2 core',
                         marker_colors=temp['operazione']), 1, 2)

    temp = pdf[(pdf['edge size'] == lato) & (pdf['cores'] == '3')]

    fig.add_trace(go.Pie(labels=temp['operazione'], values=temp['tempo'], title='3 core',
                         marker_colors=temp['operazione']), 2, 1)

    temp = pdf[(pdf['edge size'] == lato) & (pdf['cores'] == '4')]

    fig.add_trace(go.Pie(labels=temp['operazione'], values=temp['tempo'], title='4 core',
                         marker_colors=temp['operazione']), 2, 2)

    fig.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig.update_layout(
        font=dict(size=18))
    fig.show()
    fig.write_image(f'plots/percentuale{lato}.svg')
    #plotly.offline.plot(fig, filename=f'plots/percentuale{lato}.html')


for lato in ['300', '400', '500', '600', '700']:
    get_tempi(lato)

