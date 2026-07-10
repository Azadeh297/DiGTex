
def label_map_cora(data):
    label_map = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory"
    }
    
    text_labels = [label_map[int(label)] for label in data.y]

    return text_labels
def label_map_citeseer(data):
    label_map = {
         0:"Agents",
         1:"ML", 
         2:"IR", 
         3:"DB", 
         4:"HCI", 
         5:"AI"
     }
    text_labels = [label_map[int(label)] for label in data.y]

    return text_labels
def label_map_arxiv(data):
    label_map = {
    0: "cs.NA",
    1: "cs.MM",
    2: "cs.LO",
    3: "cs.CY",
    4: "cs.CR",
    5: "cs.DC",
    6: "cs.HC",
    7: "cs.CE",
    8: "cs.NI",
    9: "cs.CC",
    10: "cs.AI",
    11: "cs.MA",
    12: "cs.GL",
    13: "cs.NE",
    14: "cs.SC",
    15: "cs.AR",
    16: "cs.CV",
    17: "cs.GR",
    18: "cs.ET",
    19: "cs.SY",
    20: "cs.CG",
    21: "cs.OH",
    22: "cs.PL",
    23: "cs.SE",
    24: "cs.LG",
    25: "cs.SD",
    26: "cs.SI",
    27: "cs.RO",
    28: "cs.IT",
    29: "cs.PF",
    30: "cs.CL",
    31: "cs.IR",
    32: "cs.MS",
    33: "cs.FL",
    34: "cs.DS",
    35: "cs.OS",
    36: "cs.GT",
    37: "cs.DB",
    38: "cs.DL",
    39: "cs.DM"
}
    # data.y در ogbn-arxiv شکلی مثل (num_nodes, 1) داره، باید فلت بشه
    text_labels = [label_map[int(label)] for label in data.y.view(-1)]
    return text_labels

def label_map_pubmed(data):
    label_map = {
    0: 'Diabetes Mellitus, Experimental',
    1: 'Diabetes Mellitus Type 1', 
    2: 'Diabetes Mellitus Type 2'
    }
    text_labels = [label_map[int(label)] for label in data.y]
    return text_labels

    