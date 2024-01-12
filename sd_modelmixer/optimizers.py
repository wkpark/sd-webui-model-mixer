"""
optimizer settings and ui. based on https://github.com/Xerxemi/auto-MBW-rt by @Xerxemi

simplified by @wkpark

License: AGPL
"""
import math
import os
import gradio as gr


optimizer_params = {
    "Hill Climbing Optimizer": {
        "epsilon": 0.03,
        "distribution": ["normal", "laplace", "logistic", "gumbel"],
        "n_neighbours": 3,
    },
    "Stochastic Hill Climbing Optimizer": {
        "epsilon": 0.03,
        "distribution": ["normal", "laplace", "logistic", "gumbel"],
        "n_neighbours": 3,
        "p_accept": 0.1,
    },
    "Repulsing Hill Climbing Optimizer": {
        "epsilon": 0.03,
        "distribution": ["normal", "laplace", "logistic", "gumbel"],
        "n_neighbours": 3,
        "repulsion_factor": 5.0,
    },
    "Simulated Annealing Optimizer": {
        "epsilon": 0.03,
        "distribution": ["normal", "laplace", "logistic", "gumbel"],
        "n_neighbours": 3,
        "start_temp": 1.0,
        "annealing_rate": 0.97,
    },
    "Downhill Simplex Optimizer": {
        "alpha": 1.0,
        "gamma": 2.0,
        "beta": 0.5,
        "sigma": 0.5,
    },
    "Random Search Optimizer": {},
    "Grid Search Optimizer": {
        "step_size": 1,
    },
    "Random Restart Hill Climbing Optimizer": {
        "epsilon": 0.03,
        "distribution": ["normal", "laplace", "logistic", "gumbel"],
        "n_neighbours": 3,
        "n_iter_restart": 10,
    },
    "Random Annealing Optimizer": {
        "epsilon": 0.03,
        "distribution": ["normal", "laplace", "logistic", "gumbel"],
        "n_neighbours": 3,
        "start_temp": 10.0,
        "annealing_rate": 0.98,
    },
    "Powells Method": {
        "iters_p_dim": 10,
    },
    "Pattern Search": {
        "n_positions": 4,
        "pattern_size": 0.25,
        "reduction": 0.9,
    },
    "Parallel Tempering Optimizer": {
        "population": 10,
        "n_iter_swap": 10,
        "rand_rest_p": 0.0,
    },
    "Particle Swarm Optimizer": {
        "population": 10,
        "inertia": 0.5,
        "cognitive_weight": 0.5,
        "social_weight": 0.5,
        "rand_rest_p": 0.0,
    },
    "Spiral Optimization": {
        "population": 10,
        "decay_rate": 0.99,
    },
    "Evolution Strategy Optimizer": {
        "population": 10,
        "mutation_rate": 0.7,
        "crossover_rate": 0.3,
        "rand_rest_p": 0.0,
    },
    "Bayesian Optimizer": {
        "xi": 0.3,
        "max_sample_size": 10000000,
        "sampling": {
            "random": 1000000,
        },
        "rand_rest_p": 0.0,
    },
    "Lipschitz Optimizer": {
        "max_sample_size": 10000000,
        "sampling": {
            "random": 1000000,
        },
    },
    "Direct Algorithm": {},
    "Tree Structured Parzen Estimators": {
        "gamma_tpe": 0.2,
        "max_sample_size": 10000000,
        "sampling": {
            "random": 1000000,
        },
        "rand_rest_p": 0.0,
    },
    "Forest Optimizer": {
        "xi": 0.3,
        "tree_regressor": ["extra_tree", "random_forest", "gradient_boost"],
        "max_sample_size": 10000000,
        "sampling": {
            "random": 1000000,
        },
        "rand_rest_p": 0.0,
    },
}


def optimizer_types():
    return [*optimizer_params.keys()]


def ui_optimizers(default="None"):
    with gr.Group():
        tabs = {}
        defaults = {}
        for optimizer in optimizer_params:
            visible = default == optimizer # show if selected optimizer
            lab = optimizer.replace("Optimizer", "") + " Options"
            if len(optimizer_params[optimizer]) == 0:
                with gr.Group(visible=visible) as tab:
                    with gr.Row():
                        gr.HTML("<p>No optimizer options</p>", label=lab, show_label=False)
                    defaults[optimizer] = {}
                    tabs[optimizer] = tab
                continue
            with gr.Group(visible=visible) as tab, gr.Accordion(lab, open=True):
                tabs[optimizer] = tab
                elements = {}
                with gr.Row():
                    default = {}
                    for setting in optimizer_params[optimizer]:
                        lab = " ".join([w[0].upper() + w[1:] for w in setting.split("_")])
                        value = optimizer_params[optimizer][setting]
                        element_name = f"{setting}"
                        if type(value) is int:
                            elements[element_name] = gr.Number(label=lab, value=value, precision=0)
                            default[setting] = value
                        elif type(value) is float:
                            default[setting] = value
                            # prepare step
                            if value == 0:
                                step = 0.01
                            else:
                                step = math.log10(value)
                                remain = step - int(step)
                                step = int(step) - 1 if remain != 0 else int(step)
                                step = 10**(step - 1)
                                step = 0.01 if step > 0.01 else step

                            elements[element_name] = gr.Number(label=lab, value=value, precision=8, step=step)
                        elif type(value) is list:
                            elements[element_name] = gr.Dropdown(label=lab, choices=value, value=value[0])
                            default[setting] = value[0]
                        elif type(value) is dict:
                            # only "random" key possible
                            lab = "Random " + lab
                            v = value["random"]
                            elements[element_name] = gr.Number(label=lab, value=v, precision=0)
                            default[setting] = value
                        else:
                            elements[element_name] = gr.Textbox(label=lab, value=value)
                            default[setting] = value

                    defaults[optimizer] = default

                def update_optimizer(element, states, name, optimizer):
                    settings = states.get(optimizer, {})
                    if name == "sampling":
                        settings[name] = {"random": element}
                    else:
                        settings[name] = element

                    states.update({optimizer: settings})

                    return states

                #states = gr.State(optimizer_params)
                states = gr.State(defaults)
                for name, elem in elements.items():
                    elem.change(
                        fn=(lambda n, o: lambda elem, states: update_optimizer(elem, states, n, o))(name, optimizer),
                        inputs=[elem, states],
                        outputs=[states],
                    )

    return tabs, states
