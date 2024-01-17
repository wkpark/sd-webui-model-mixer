"""
hyperactive optimizer for model-mixer

based on auto-MBW-rt. heavily modified by wkpark
"""
import json
import numpy as np
import os
import re
import statistics
import sys

from modules import shared, images
from modules import txt2img
from pathlib import Path
from PIL import Image

from .classifier import get_classifiers, classifier_score
from .optimizers import optimizer_types
from .utils import all_blocks, _all_blocks, load_module


classifiers = get_classifiers()

def para_to_weights(para, weights=None, isxl=False):
    BLOCKS = all_blocks(isxl)
    BLOCKLEN = (12 if not isxl else 9)*2 + 2

    weights = {} if weights is None else dict(zip(range(len(weights)), weights))
    for k in para:
        name = k.split(".")
        modelidx = ord(name[0].split("_")[1]) - 98
        weight = weights.get(modelidx, [0.0]*BLOCKLEN)
        j = BLOCKS.index(name[1])
        weight[j] = para[k]
        weights[modelidx] = weight

    maxid = max(weights.keys())
    nweights = [""] * (maxid + 1)
    for i in weights.keys():
        nweights[i] = ",".join([("0" if float(f) == 0.0 else str(f)) for f in weights[i]])

    return nweights


def normalize_mbw(mbw, isxl):
    """Normalize Merge Block Weights"""
    MAXLEN = 26 - (0 if not isxl else 6)
    BLOCKLEN = 12 - (0 if not isxl else 3)
    
    # no mbws blocks selected or have 'ALL' alias
    if len(mbw) == 0 or 'ALL' in mbw:
        # select all blocks
        mbw = [ 'BASE', 'INP*', 'MID', 'OUT*' ]

    # fix alias
    if 'MID' in mbw:
        i = mbw.index('MID')
        mbw[i] = 'M00'

    # expand some aliases
    if 'INP*' in mbw:
        for i in range(BLOCKLEN):
            name = f"IN{i:02d}"
            if name not in mbw:
                mbw.append(name)
    if 'OUT*' in mbw:
        for i in range(BLOCKLEN):
            name = f"OUT{i:02d}"
            if name not in mbw:
                mbw.append(name)

    BLOCKS = all_blocks(isxl)

    sort = []
    for b in BLOCKS[:MAXLEN]:
        if b in mbw:
            sort.append(b)

    return sort


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text


# from modules/generation_parameters_copypaste.py
re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
def parse(lastline):
    """from parse_generation_parameters(x: str)"""
    res = {}
    for k, v in re_param.findall(lastline):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

            res[k] = v
        except Exception:
            print(f"Error parsing \"{k}: {v}\"")

    return res


def tally_score(tally_type, imagescores):
    if tally_type == "Arithmetic Mean":
        testscore = statistics.mean(imagescores)
    elif tally_type == "Geometric Mean":
        testscore = statistics.geometric_mean(imagescores)
    elif tally_type == "Harmonic Mean":
        testscore = statistics.harmonic_mean(imagescores)
    elif tally_type == "Quadratic Mean":
        testscore = np.sqrt(np.mean(np.array(imagescores)**2))
    elif tally_type == "Cubic Mean":
        testscore = np.cbrt(np.mean(np.array(imagescores)**3))
    elif tally_type == "A/G Mean":
        testscore = (statistics.mean(imagescores)/statistics.geometric_mean(imagescores))*statistics.mean(imagescores)
    elif tally_type == "G/H Mean":
        testscore = (statistics.geometric_mean(imagescores)/statistics.harmonic_mean(imagescores))*statistics.mean(imagescores)
    elif tally_type == "A/H Mean":
        testscore = (statistics.mean(imagescores)/statistics.harmonic_mean(imagescores))*statistics.mean(imagescores)
    elif tally_type == "Median":
        testscore = statistics.median(imagescores)
    elif tally_type == "Min":
        testscore = min(imagescores)
    elif tally_type == "Max":
        testscore = max(imagescores)
    elif tally_type == "Mid-Range":
        testscore = (min(imagescores)+max(imagescores))/2

    return testscore


def get_payloads_from_path(path):
    if path is None or path == "None" or path == "":
        return []

    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.exists(path):
        path = os.path.join(paths.data_path, path) # XXX
    image_dir = Path(path)
    images_found = list(image_dir.glob(f"*.png"))

    payloads = []
    if len(images_found) > 0:
        for pngpath in images_found:
            image = Image.open(pngpath)
            geninfo, _ = images.read_info_from_image(image)

            *lines, lastline = geninfo.strip().split("\n")
            excluded = []
            while "Steps" not in lastline:
                excluded.insert(0, lastline)
                lastline = lines.pop()

            prompt_parts = "\n".join(lines).split('Negative prompt:')
            prompt, negative_prompt = [s.strip() for s in prompt_parts[:2] + ['']*(2-len(prompt_parts))]

            res = parse(lastline)
            payload = {"prompt": prompt, "neg_prompt": negative_prompt}
            if "Seed" in res:
                payload.update({"Seed": res["Seed"]})
            payloads.append(payload)

    return payloads


display_images = None
def hyper_optimizer(
        txt2img_args=None,
        seed_index=0,
        passes=1,
        classifier="score_image_reward",
        payload_path=None,
        search_type_a="Hill Climbing Optimizer", search_type_b="None", search_balance=0,
        tally_type="Arithmetic Mean",
        search_iterations=250, search_time=60,
        variable_blocks=None,
        variable_models=None,
        search_upper=0.2, search_lower=-0.2, search_max=1.0,
        steps_or_inc=5,
        initialize_grid=4, initialize_vertices=4, initialize_random=2,
        warm_start=True,
        enable_early_stop=False, n_iter_no_change=25, tol_abs=0, tol_rel=0,
        search_opts_a=None, search_opts_b=None):

    import inspect
    prompt_idx = 1 # 1 for webui 1.7.0, 2 for webui dev. determined later

    if steps_or_inc <= 0:
        steps_or_inc = 5
        print(" - set steps_or_inc as 5")
    steps_or_inc = round(steps_or_inc) if steps_or_inc >= 1 else steps_or_inc

    if search_upper == search_lower:
        search_lower = -search_upper
    if search_upper < search_lower:
        search_upper, search_lower = search_lower, search_upper
    print(" - set search lower, upper =", search_lower, search_upper)

    check_args = inspect.signature(txt2img.txt2img).parameters
    _prompt = list(check_args)[1]
    if ': str' not in str(check_args[_prompt]):
        # new webui dev calling convension. fix request arg order
        _req = txt2img_args.pop(23) # 23-th argument is 'gr.Request'
        txt2img_args.insert(1, _req) # insert request to the 2nd argument
        prompt_idx = 2
        print(" - fix request parameter order...")


    def score_func(classifier, image, prompt):
        if classifier not in classifiers:
            raise ValueError("no classifier found")

        module_path = classifiers[classifier]
        if module_path and image:
            score = classifier_score(module_path, image, prompt)
            return score

        raise ValueError("no image or no classifier found")


    def hyper_score(localargs):
        global display_images

        tunables = localargs.pass_through["tunables"]
        isxl = localargs.pass_through["isxl"]
        uses = localargs.pass_through["uses"]
        usembws = localargs.pass_through["usembws"]
        testweights = localargs.pass_through["weights"].copy()
        prompt = localargs.pass_through["prompt"]
        payload_path = localargs.pass_through["payload_path"]
        seed_index = localargs.pass_through["seed_index"]

        BLOCKS = all_blocks(isxl)
        payloads = get_payloads_from_path(payload_path)

        if shared.state.interrupted:
            raise ValueError("Error: Interrupted!")

        testalpha = [""] * len(usembws)
        # gather tunable variables into override weights
        for k in tunables:
            name = k.split(".")
            modelidx = ord(name[0].split("_")[1]) - 98
            if uses[modelidx] is False:
                continue

            if len(usembws[modelidx]) == 0:
                testalpha[modelidx] = localargs[k]
                continue

            weight = testweights[modelidx]
            j = BLOCKS.index(name[1])
            weight[j] = localargs[k]
            testweights[modelidx] = weight

        _weights = [""] * len(testweights)
        for j in range(len(testweights)):
            _weights[j] = ','.join([("0" if float(w) == 0.0 else str(w)) for w in testweights[j]])

        print(" - test weights: ", _weights)
        print(" - test alphas: ", testalpha)

        # setup override weights. will be replaced with mm_weights
        shared.modelmixer_overrides = {"weights": _weights, "alpha": testalpha, "uses": uses}

        if len(payloads) == 0:
            images = []
            ret = txt2img.txt2img(*txt2img_args)
            if len(ret[0]) > 0:
                # get image from output gallery
                score = score_func(classifier, ret[0][0], prompt)
                images.append(ret[0][0])
        else:
            images = []
            scores = []
            # generate images
            args = txt2img_args.copy()
            for payload in payloads:
                args[prompt_idx] = payload["prompt"]
                args[prompt_idx + 1] = payload["neg_prompt"]

                orig_seed = None
                if "Seed" in payload:
                    ii = seed_index + 1
                    print(" - Overriden seed =", payload["Seed"])
                    args[ii] = payload["Seed"]

                ret = txt2img.txt2img(*args)

                if len(ret[0]) > 0:
                    score = score_func(classifier, ret[0][0], args[prompt_idx])
                    images.append(ret[0][0])
                    scores.append(score)

            tally_type = localargs.pass_through["tally_type"]
            score = tally_score(tally_type, scores)

        display_images = images
        print(" - score is ", score)

        return score

    if txt2img_args[prompt_idx] is None or txt2img_args[prompt_idx].strip() == "":
        raise ValueError("FATAL: Empty prompt payload!")

    # get initial positions
    initial = getattr(shared, "_optimizer_config", None)
    current = getattr(shared, "modelmixer_config", None)
    if current is None or initial is None:
        # if current is None, prepare merged model
        if current is None:
            initial = None
            txt2img.txt2img(*txt2img_args)
            current = getattr(shared, "modelmixer_config", None)

        assert current is not None

        if initial is None:
            shared._memory_warm_start = None
            shared._memory_warm_hash = None
            shared._optimizer_config = current
            initial = current

    # import hyperactive
    import hyperactive.optimizers
    from hyperactive import Hyperactive
    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    hyper = Hyperactive(n_processes=1, distribution="joblib")

    uses = initial["uses"] # used models
    weights = initial["weights"] # normalized weights
    usembws = initial["usembws"] # merged blocks
    alpha = initial["alpha"] # alpha values without merged blocks
    selected_blocks = initial["selected"]

    isxl = shared.sd_model.is_sdxl

    blocks = _all_blocks(isxl)
    _BLOCKS = all_blocks(isxl)

    print("#"*20," Auto merger using Hyperactive ", "#"*20)
    
    # setup search space
    search_space = {}
    if variable_blocks is not None and len(variable_blocks) > 0:
        variable_blocks = normalize_mbw(variable_blocks, isxl)
    else:
        variable_blocks = None

    if variable_models is not None and len(variable_models) == 0:
        variable_models = None

    # setup override uses
    override_uses = uses.copy()

    k = 0
    for i in range(len(uses)):
        if uses[i] is not True:
            continue

        # is this model in the variable_models? e.g.) chr(0+66) == B
        if variable_models is not None:
            if chr(i+66) not in variable_models:
                override_uses[i] = False
                k += 1
                continue

        name = f"model_{chr(i + 98)}"
        if len(usembws[k]) == 0:
            # no merged block weighs
            val = alpha[k]
            # setup range, lower + val ~ val + upper < search max. e.g.) -0.3 + val ~ val + 0.3 < 0.5
            lower = max(val + search_lower, 0)
            upper = min(val + search_upper, search_max)
            if steps_or_inc >= 1:
                search_space[f"{name}.alpha"] = [*np.round(np.linspace(lower, upper, steps_or_inc), 8)]
            elif steps_or_inc < 1:
                search_space[f"{name}.alpha"] = [*np.round(np.arange(lower, upper, steps_or_inc), 8)]

            k += 1
            continue

        weight = weights[k]
        mbw = normalize_mbw(usembws[k], isxl)
        for b in selected_blocks:
            j = blocks.index(b)
            if j < len(weight) and _BLOCKS[j] in mbw:
                if variable_blocks is not None and _BLOCKS[j] not in variable_blocks:
                    continue
                val = weight[j]
                # setup range, lower + val ~ val + upper < search max. e.g.) -0.3 + val ~ val + 0.3 < 0.5
                lower = max(val + search_lower, 0)
                upper = min(val + search_upper, search_max)
                if steps_or_inc >= 1:
                    search_space[f"{name}.{_BLOCKS[j]}"] = [*np.round(np.linspace(lower, upper, steps_or_inc), 8)]
                elif steps_or_inc < 1:
                    search_space[f"{name}.{_BLOCKS[j]}"] = [*np.round(np.arange(lower, upper, steps_or_inc), 8)]
        k += 1

    print(" - search_space keys =", search_space.keys())

    # setup warm_start
    if warm_start:
        current = getattr(shared, "modelmixer_config", None)

        warm = {}
        _uses = current["uses"] # used models
        _weights = current["weights"] # normalized weights
        _usembws = current["usembws"] # merged blocks
        _alpha = current["alpha"] # merged blocks
        _selected_blocks = current["selected"]
        k = 0 # fix index for not used model. e.g.) A, B, E (C is not selected case)
        for i in range(len(_uses)):
            if _uses[i] is not True:
                continue

            # is this model in the variable_models? e.g.) chr(0+66) == B
            if variable_models is not None and chr(i+66) not in variable_models:
                k += 1
                continue

            name = f"model_{chr(i + 98)}"
            if len(_usembws[k]) == 0:
                # no merged block weighs
                val = _alpha[k]
                warm[f"{name}.alpha"] = val

                k += 1
                continue

            weight = _weights[k]
            mbw = normalize_mbw(_usembws[k], isxl)
            for b in _selected_blocks:
                j = blocks.index(b)
                if j < len(weight) and _BLOCKS[j] in mbw:
                    val = weight[j]
                    warm[f"{name}.{_BLOCKS[j]}"] = val
            k += 1

        print(" - warm_start = ", warm)
        warm_start = [warm]

    prompt = txt2img_args[prompt_idx]

    # check warm_start hash
    warm_start_hash_args = [
        classifier,
        search_type_a, search_type_b, search_balance,
        tally_type,
        search_iterations,
        variable_blocks,
        variable_models,
        search_upper, search_lower, search_max,
        initialize_grid, initialize_vertices, initialize_random,
        search_opts_a, search_opts_b]
    warm_hash = hash(json.dumps(warm_start_hash_args))

    warm_hash_saved = getattr(shared, "_memory_warm_hash", None)
    if warm_hash is not None and warm_hash == warm_hash_saved:
        memory_warm_start = getattr(shared, "_memory_warm_start", None)
    else:
        memory_warm_start = None

    for _pass in range(passes):
        if _pass > 0:
            warm_start = []

        early_stopping = {
            "n_iter_no_change": n_iter_no_change,
            "tol_abs": tol_abs,
            "tol_rel": tol_rel
        } if enable_early_stop else None

        pass_through = {
            "tunables": [*search_space.keys()],
            "weights": weights,
            "alpha": alpha,
            "uses": override_uses,
            "usembws": usembws,
            "classifier": classifier,
            "payload_path": payload_path,
            "tally_type": tally_type,
            "isxl": isxl,
            "prompt": prompt,
            "seed_index": seed_index,
            #"seedplus": seedplus,
        }
        search_type_a = search_type_a.replace(" ", "")
        search_type_b = search_type_b.replace(" ", "")
        print(" - search type = ", search_type_a, search_type_b)
        print(search_opts_a, search_opts_b)

        # optimizer strategy to combine 2 opts
        # we set search_iterations min to 10 and step of search_balance to 0.1 since hyperactive/pandas has an odd tolerance to low iterations during use of dual optimizers
        search_type_a = getattr(hyperactive.optimizers, search_type_a) if search_type_a != "None" else None
        search_type_b = getattr(hyperactive.optimizers, search_type_b) if search_type_b != "None" else None
        if search_balance != 0 and search_balance != 1 and search_type_b is not None:
            opt_strat = CustomOptimizationStrategy()
            opt_strat.add_optimizer(search_type_a(**search_opts_a), duration=round(1 - search_balance, 1))
            opt_strat.add_optimizer(search_type_b(**search_opts_b), duration=search_balance)
        elif search_balance == 0:
            opt_strat = search_type_a()
        elif search_balance == 1 and search_type_b is not None:
            opt_strat = search_type_b(**search_opts_b)
        else:
            opt_strat = search_type_a(**search_opts_a)

        print(" - opt_strategy = ", opt_strat, type(opt_strat))

        hyper.opt_pros.pop(0, None)
        # memory=True here or else memory defaults to "share",
        # spawning a mp.Manager() that goes rogue due to webui os._exit(0) on SIGINT

        hyper.add_search(
            hyper_score,
            search_space,
            optimizer=opt_strat,
            n_iter=search_iterations,
            n_jobs=1,
            initialize={"grid": initialize_grid, "vertices": initialize_vertices, "random": initialize_random, "warm_start": warm_start},
            pass_through=pass_through,
            early_stopping=early_stopping,
            memory=True,
            memory_warm_start=memory_warm_start
        )

        # run main optimizer
        shared.state.begin(job="modelmixer-auto-merger")
        best_para = None
        try:
            hyper.run(search_time*60)
            best_para = hyper.best_para(hyper_score)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            shared.state.end()

        if best_para is not None:
            best_weights = para_to_weights(best_para, weights, isxl)
            print(" - Best weights para = ", best_weights, override_uses)

            # setup override weights. will be replaced with mm_weights

            shared.modelmixer_overrides = {"weights": best_weights, "uses": override_uses}

            # generate image with the optimized parameter
            ret = None
            ret = txt2img.txt2img(*txt2img_args)
            if ret and ret[0] is not None:
                gallery = ret[0] # gallery

                score = score_func(classifier, gallery[0], prompt)
                print("Result score =", score)

            shared._memory_warm_start = hyper.search_data(hyper_score)
            shared._memory_warm_hash = warm_hash
            msg = "merge completed."
        else:
            shared._memory_warm_start = None
            shared._memory_warm_hash = None
            msg = "Failed to call hyper.run()"

    # search data save
    #timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M%p-%S")
    #collector = SearchDataCollector(os.path.join(folder_path, f"{model_O}-{pass}-{timestamp}.csv"))
    #collector.save(hyper.search_data(hyper_score, times=True))

    if hasattr(shared, "modelmixer_overrides"):
        delattr(shared, "modelmixer_overrides")

    return msg
