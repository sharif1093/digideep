""" https://github.com/pytorch/tnt
BSD 3-Clause License

Copyright (c) 2017- Sergey Zagoruyko,
Copyright (c) 2017- Sasank Chilamkurthy, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""



import numpy as np
import visdom
from digideep.utility.visdom_engine.Instance import VisdomInstance

class VisdomWrapper(object):
    """
        This class does not need to be serializable.
        Can be given among the `kwargs` params:
        
        * 'opts'
        * 'env'

        If you want to be consistent between different runs,
        you must use assign 'win' as input.


        For example:
            v = VisdomWrapper('line', win='TestLoss', opts={'title':'TestLoss'}, X=np.array([1]), Y=np.array([4]))
    """
    def __init__(self, command, win, **kwargs):
        self.viz = VisdomInstance.getVisdomInstance()
        self.command_fcn = getattr(self.viz, command)
        self.state = {'command':command, 'win':win, 'kwargs_static':{**kwargs}}    
    def run(self, **kwargs):
        if self.state['win'] is None:
            self.state['win'] = self.command_fcn(
                **kwargs,
                **self.state['kwargs_static']
            )
        else:
            self.command_fcn(
                **kwargs,
                **self.state['kwargs_static'],
                win=self.state['win']
            )
    def get_win(self):
        return self.state['win']
    def get_env(self):
        if 'env' is self.state["kwargs_static"]:
            return self.state["kwargs_static"]['env']
        else:
            return "main"
    

class VisdomWrapperPlot(VisdomWrapper):
    """
    In the append function, user should provide X=np.array(...), Y=np.array(...)
    """
    def __init__(self, plot_type, win, **kwargs):
        super(VisdomWrapperPlot, self).__init__(plot_type, win=win, **kwargs)
    def append(self, **kwargs):
        if 'X' in kwargs:
            kwargs['X'] = self._prepare_data(kwargs['X'])
        if 'Y' in kwargs:
            kwargs['Y'] = self._prepare_data(kwargs['Y'])
        if 'Z' in kwargs:
            kwargs['Z'] = self._prepare_data(kwargs['Z'])
        
        if self.state['win'] is None:
            self.run(**kwargs)
        else:
            self.run(update='append', **kwargs)
    def _prepare_data(self, data):
        if (not isinstance(data, np.ndarray)) or (isinstance(data, np.ndarray) and data.ndim == 0):
            return np.array([data])
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise Exception('The data type for X|Y|Z specified for updating the plot is not appropriate.')

# For properties, we need something bidirectional.
# If a value is changed elsewhere, visdom should
# noticed and update the value. If the value is
# changed inside visdom, other elements should get
# notice about the change instantly.
# class VisdomWrapperProperty(VisdomWrapper):
#     def __init__(self, name, ptype, value):
# ---------------------------------------------------
#     properties = [
#         {'type': 'text', 'name': 'Text input', 'value': 'initial'},
#         {'type': 'number', 'name': 'Number input', 'value': '12'},
#         {'type': 'button', 'name': 'Button', 'value': 'Start'},
#         {'type': 'checkbox', 'name': 'Checkbox', 'value': True},
#         {'type': 'select', 'name': 'Select', 'value': 1, 'values': ['Red', 'Green', 'Blue']},
#     ]

#     properties_window = viz.properties(properties)

#     def properties_callback(event):
#         if event['event_type'] == 'PropertyUpdate':
#             prop_id = event['propertyId']
#             value = event['value']
#             if prop_id == 0:
#                 new_value = value + '_updated'
#             elif prop_id == 1:
#                 new_value = value + '0'
#             elif prop_id == 2:
#                 new_value = 'Stop' if properties[prop_id]['value'] == 'Start' else 'Start'
#             else:
#                 new_value = value
#             properties[prop_id]['value'] = new_value
#             viz.properties(properties, win=properties_window)
#             viz.text("Updated: {} => {}".format(properties[event['propertyId']]['name'], str(event['value'])),
#                      win=callback_text_window, append=True)

#     viz.register_event_handler(properties_callback, properties_window)


# Unit Test
if __name__ == '__main__':
    import time
    from VisdomEngine.WebServer import VisdomWebServer
    
    # We can get it from comman-line:
    port = 8097
    enable_login = False

    VisdomWebServer(port=port, enable_login=enable_login)
    VisdomInstance(port=port, log_to_filename='vizdom.log', replay=True)

    # Assigning windows is very necessary if we want to replay in the same window.
    v1 = VisdomWrapperPlot('line', win='LossWindow')
    for i in range(500):
        v1.append(X=np.random.rand(),Y=np.random.rand(), name='1')
        v1.append(X=np.random.rand()+1,Y=np.random.rand(), name='2')
        time.sleep(0.02)


    v2 = VisdomWrapperPlot('line',
        win='AreaPlot', 
        opts=dict(
            fillarea=True,
            showlegend=False,
            width=800,
            height=800,
            xlabel='Time',
            ylabel='Volume',
            ytype='log',
            title='Stacked area plot',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
        )
    )

    Y = np.linspace(0, 4, 200)
    v2.append(Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)), X=np.column_stack((Y, Y)))
    v2.append(Y=np.sqrt(Y) + 1, X=Y, name='3')

    input('Waiting for callbacks, press enter to quit.')

