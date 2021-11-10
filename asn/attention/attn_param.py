from asn.conversion.utils import predict_model_length

def set_attn_param(layer_idx= None,
                   Ax1=None, Ax2=None, AxWidth1=None, AxWidth2=None, Apeak=2, Abase=1, Ashape='oval', exponent=None,
                   IxWidth=0,
                   InputGain=0, OutputGain=0, Precision=0):

    """ Function to set-up attention parameter for a given layer # Todo
    :parameter
    Layer: idx of layer to be targeted in test model
    Ax1: Vertical center of attention field (rows), if list various mf_mats will be made
    Ax2: Horizontal center of attention field (columns), if list various mf_mats will be made
    AxWidth1: vertical extent/width  attention field
    AxWidth2: horizontal extent/width  attention field
    Apeak: peak amplitude of attention field
    Abase: baseline of attention field for unattended locations/features
    Ashape: either 'oval' or 'cross'
    IxWidth: Width of kernel for suppressive kernel
    exponent: exponent on attention field

    InputGain: Incoming current will be multiplied with 1+ R*Inputgain
    OutputGain: Outgoing synaptic connections will be scaled by 1+ R*Outputgain
    Precision: Precision distribution will be scaled.

    :returns
    attn_param: dict with attentional params
    """

    if isinstance(Ax1, list) & isinstance(Ax2, list):
        if not len(Ax1) == len(Ax2):
            raise ValueError('Centre coordinates have non-matching dimensions with '+ str(len(Ax1)) + 'and' +
                             str(len(Ax2)))

    attn_param = {'layer_idx':layer_idx,
                  'Ax1': Ax1,
                  'Ax2': Ax2,
                  'AxWidth1': AxWidth1,
                  'AxWidth2': AxWidth2,
                  'Apeak': Apeak,
                  'Abase': Abase,
                  'Ashape': Ashape,
                  'IxWidth': IxWidth,
                  'exponent': exponent,
                  'InputGain': InputGain,
                  'OutputGain': OutputGain,
                  'Precision': Precision}

    return attn_param


def set_model_attn_param(model, *args):
    """
    :param model: analog model
    :param **kwargs: dict with one or more attn_param dicts
    :return: Model wide attentional settings
    """
    model_length = predict_model_length(model)
    # Preallocate dict
    model_attn_params = {i: set_attn_param() for i in range(model_length)}

    for dict in args[0][0]:
        print('Integrating these attentional modulations: ')
        print(args[0][0][dict])
        model_attn_params[args[0][0][dict]['layer_idx']] = args[0][0][dict]

    return model_attn_params

