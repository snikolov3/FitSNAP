import os, json, time, math
import numpy as np

class CostObject:
    # @snap.pt.rank_zero
    def __init__(self):
        self.conts = None
        self.cost = 999999999.9999
        self.unweighted = np.array([])
        self.weights = np.array([])

    # @snap.pt.rank_zero
    def cost_contributions(self,cost_conts,weights):
        cost_conts = np.array(cost_conts) 
        weights = np.array(weights)
        wc_conts = weights*cost_conts
        self.unweighted = cost_conts
        self.weights = weights
        self.conts = wc_conts

    # @snap.pt.rank_zero
    def add_contribution(self,costi,weighti=1.0):
        try:
            if self.conts == None:
                cond = True
            else:
                cond = False
        except ValueError:
            cond = False
        if cond:
            cost_conts = np.array([costi])
            weights = np.array([weighti])
            wc_conts = weights*cost_conts
            self.conts = wc_conts
            self.weights = np.append(self.weights, weighti)
            self.unweighted = np.append(self.unweighted, costi)
        else:
            self.conts = np.append(self.conts,costi*weighti)
            self.weights = np.append(self.weights,weighti)
            self.unweighted = np.append(self.unweighted,costi)

    # @snap.pt.rank_zero
    def evaluate_cost(self):
        costi = np.sum(self.conts)
        self.cost = costi
        return costi


class HyperparameterStruct:
    # @snap.pt.rank_zero
    def __init__(self,
            ne,
            nf,
            ns,
            eranges,
            ffactors,
            sfactors,):
        self.ne = ne
        self.nf = nf
        self.ns = ns
        self.nh = ne+nf+ns
        self.erangesin = eranges
        self.ffactorsin = ffactors
        self.sfactorsin = sfactors
        self.set_eranges()
        self.set_ffactors()
        self.set_sfactors()

    # @snap.pt.rank_zero
    def set_eranges(self):
        if len(self.erangesin) != self.ne and len(self.erangesin) == 1:
            self.eranges = self.erangesin * self.ne
        elif len(self.erangesin) == self.ne:
            self.eranges = self.erangesin
        else:
            raise ValueError('incorrect number of values for energy group weight ranges, ether specify range for each group or sepecify one range to be applied to all groups')

    # @snap.pt.rank_zero
    def set_ffactors(self):
        if len(self.ffactorsin) != self.nf and len(self.ffactorsin) == 1:
            self.ffactors = self.ffactorsin * self.nf
        elif len(self.ffactorsin) == self.nf:
            self.ffactors = self.ffactorsin
        else:
            raise ValueError('incorrect number of values for force group weight ratios, ether specify range for each group or sepecify one range to be applied to all groups')
    
    # @snap.pt.rank_zero
    def set_sfactors(self):
        if len(self.sfactorsin) != self.ns and len(self.sfactorsin) == 1:
            self.sfactors = self.sfactorsin * self.ns
        elif len(self.sfactorsin) == self.ns:
            self.sfactors = self.sfactorsin
        else:
            raise ValueError('incorrect number of values for stress group weight ratios, ether specify range for each group or sepecify one range to be applied to all groups')

    # @snap.pt.rank_zero
    def random_params(self,inputseed=None):
        if inputseed != None:
            np.random.seed(inputseed)
        self.eweights = np.random.rand(self.ne) * np.array([np.random.choice(self.eranges[ihe]) for ihe in range(self.ne)])
        f_factors =  np.random.rand(self.nf) * np.array([np.random.choice(self.ffactors[ihf]) for ihf in range(self.nf)])
        self.fweights = self.eweights * f_factors
        s_factors =  np.random.rand(self.ns) * np.array([np.random.choice(self.sfactors[ihs]) for ihs in range(self.ns)])
        self.fweights = self.eweights * s_factors
        return np.concatenate((self.eweights,self.fweights, self.sweights))

    # @snap.pt.rank_zero
    def lhs_params(self,num_samples, inputseed=None):
        if inputseed != None:
            np.random.seed(inputseed)
        variable_ranges_dicti = {}
        variable_types_dict = {}
        for i in range(self.ne):
            # print (i,'ew,fr', [np.log10(min(self.eranges[0])),np.log10(max(self.eranges[0]))], [np.log10(min(self.ffactors[0])),np.log10(max(self.ffactors[0]))])
            #variable_ranges_dict['ew%d'%i] = [min(self.eranges),max(self.eranges)]
            variable_ranges_dicti['ew%d'%i] = [float(np.log10(min(self.eranges[0]))),float(np.log10(max(self.eranges[0])))]
            #variable_ranges_dict['fr%d'%i] = [min(self.ffactors),max(self.ffactors)]
            variable_ranges_dicti['fr%d'%i] = [float(np.log10(min(self.ffactors[0]))),float(np.log10(max(self.ffactors[0])))]
            #variable_ranges_dict['sr%d'%i] = [min(self.sfactors),max(self.sfactors)]
            variable_ranges_dicti['sr%d'%i] = [float(np.log10(min(self.sfactors[0]))),float(np.log10(max(self.sfactors[0])))]
            #variable_types_dict['ew%d'%i] = float
            variable_types_dict['ew%d'%i] = 'logfloat'
            #variable_types_dict['fr%d'%i] = float
            variable_types_dict['fr%d'%i] = 'logfloat'
            #variable_types_dict['sr%d'%i] = float
            variable_types_dict['sr%d'%i] = 'logfloat'
        # print ("HH varrange dict: ",variable_ranges_dicti)
        lhsamples = latin_hypercube_sample(variable_ranges_dicti, variable_types_dict, num_samples)
        return lhsamples

# tournament selection
# @snap.pt.rank_zero
def tournament_selection(population, scores, k=3, inputseed=None):
    if inputseed != None:
        np.random.seed(inputseed)
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

class Selector:
    # @snap.pt.rank_zero
    def __init__(self,selection_style = 'tournament'):
        self.selection_style = selection_style
        self.set_selector()

    # @snap.pt.rank_zero
    def set_selector(self):
        if self.selection_style == 'tournament':
            self.selection = tournament_selection

# when 2 parent creatures love eachother very much, they make/adopt 2 children creatures
# @snap.pt.rank_zero
def crossover(p1, p2, ne, w_combo_delta=np.array([]), ef_rat_delta=np.array([]), es_rat_delta=np.array([]), inputseed=None):
    if inputseed != None:
        np.random.seed(inputseed)
    c1, c2 = p1.copy(), p2.copy()
    c1e,c1f,c1s = tuple(c1.reshape(3,ne))
    c2e,c2f,c2s = tuple(c2.reshape(3,ne))
    p1e,p1f,p1s = tuple(p1.reshape(3,ne))
    p2e,p2f,p2s = tuple(p2.reshape(3,ne))

    # select crossover point that corresponds to a certain group
    # NOTE meg changed var name 'pt' to 'cpt' to avoid confusion with parallel tools "pt" later
    ##LOGAN QUESTION: Can you confirm these indices are correct?
    cpt = np.random.randint(1, ne-2)
    # only perform crossover between like hyperparameters (energy and energy then force and force, etc.)
    if np.shape(w_combo_delta)[0] != 0:
        c1e = np.append(p1e[:cpt], p2e[cpt:])*w_combo_delta
        c1f = np.append(p1f[:cpt], p2f[cpt:])*ef_rat_delta
        c1s = np.append(p1s[:cpt], p2s[cpt:])*es_rat_delta
        c2e = np.append(p2e[:cpt], p1e[cpt:])*w_combo_delta
        c2f = np.append(p2f[:cpt], p1f[cpt:])*ef_rat_delta
        c2s = np.append(p2s[:cpt], p1s[cpt:])*es_rat_delta
    else:
        c1e = np.append(p1e[:cpt] , p2e[cpt:])
        c1f = np.append(p1f[:cpt] , p2f[cpt:])
        c1s = np.append(p1s[:cpt] , p2s[cpt:])
        c2e = np.append(p2e[:cpt] , p1e[cpt:])
        c2f = np.append(p2f[:cpt] , p1f[cpt:])
        c2s = np.append(p2s[:cpt] , p1s[cpt:])
    c1 = np.concatenate((c1e,c1f,c1s))
    c2 = np.concatenate((c2e,c2f,c2s))
    return [c1, c2]

# @snap.pt.rank_zero
def update_weights(snap, test_w_combo, test_ef_rat, test_es_rat, gtks, size_b, grouptype, initial_weights=False):
    if initial_weights:
        if len(test_es_rat) == 1:
            tstwdct = {gkey:{'eweight':initial_weights[gkey][0]*test_w_combo[ig], 'fweight':initial_weights[gkey][1]*test_w_combo[ig]*test_ef_rat[ig], \
                             'vweight':initial_weights[gkey][2]*test_w_combo[ig]*test_es_rat[0]} for ig,gkey in enumerate(gtks)  }
        elif len(test_es_rat) == len(test_ef_rat):
            tstwdct = {gkey:{'eweight':initial_weights[gkey][0]*test_w_combo[ig], 'fweight':initial_weights[gkey][1]*test_w_combo[ig]*test_ef_rat[ig], \
                             'vweight':initial_weights[gkey][2]*test_w_combo[ig]*test_es_rat[ig]} for ig,gkey in enumerate(gtks)  }
        else:
            raise IndexError("not enough virial indices per energy and force indices")
    else:
        if len(test_es_rat) == 1:
            tstwdct = {gkey:{'eweight':test_w_combo[ig], 'fweight':test_w_combo[ig]*test_ef_rat[ig], 'vweight':test_w_combo[ig]*test_es_rat[0]} for ig,gkey in enumerate(gtks)  }
        elif len(test_es_rat) == len(test_ef_rat):
            tstwdct = {gkey:{'eweight':test_w_combo[ig], 'fweight':test_w_combo[ig]*test_ef_rat[ig], 'vweight':test_w_combo[ig]*test_es_rat[ig]} for ig,gkey in enumerate(gtks)  }
        else:
            raise IndexError("not enough virial indices per energy and force indices")

    #loop through data and update pt shared array based on group type
    for index_b in range(size_b):
        gkey = grouptype[index_b]
        if snap.pt.fitsnap_dict['Row_Type'][index_b] == 'Energy':
            snap.pt.shared_arrays['w'].array[index_b] = tstwdct[gkey]['eweight']
        elif snap.pt.fitsnap_dict['Row_Type'][index_b] == 'Force':
            snap.pt.shared_arrays['w'].array[index_b] = tstwdct[gkey]['fweight']
        elif snap.pt.fitsnap_dict['Row_Type'][index_b] == 'Stress':
            snap.pt.shared_arrays['w'].array[index_b] = tstwdct[gkey]['vweight']

# @snap.pt.rank_zero
def ediff_cost(snap, fit, g1, g2, target, grouptype, rowtype):
    # provides 'cost' for energy differences for amatrix entries:
    #  (\beta * a1 - \beta * a2) - target
    # where target is some target energy difference between structures from group 1 and group 2
    # NOTE that a1 and a2 Amatrix entries are identified by indexing. You 
    # MUST put only one structure per group to use in this objective function
    indexg1 = grouptype.index(g1)
    indexg2 = grouptype.index(g2)
    assert rowtype[indexg1] == 'Energy',"not accessing energy row for group %s" % g1
    assert rowtype[indexg2] == 'Energy',"not accessing energy row for group %s" % g2
    a1 = snap.pt.shared_arrays['a'].array[indexg1].copy()
    a2 = snap.pt.shared_arrays['a'].array[indexg2].copy()

    e1 = np.sum(a1*fit)
    e2 = np.sum(a2*fit)
    diff = e1-e2

    #snap.pt.single_print(e1,e2,diff,target)

    return np.abs(diff - target)

#NOTE fit_and_cost will likely need to be modified to print current fit if
# other objective functions are to be added. 
# @snap.pt.rank_zero
def fit_and_cost(snap,costweights):
    etot_weight, ftot_weight, stot_weight = tuple(costweights)
    #clear old fit and solve test fit
    snap.solver.fit = None
    snap.perform_fit()
    fittst = snap.solver.fit
    errstst = snap.solver.errors
    rmse_tst = errstst.iloc[:,2].to_numpy()
    #rmse_countstst = errstst.iloc[:,0].to_numpy()  ##LOGAN NOTE: unused, can just remove unless here for instructive purposes

    #snap.pt.single_print(errstst)
    rmse_eattst, rmse_fattst, rmse_sattst = rmse_tst[0:3]
    CO = CostObject()
    CO.add_contribution(rmse_eattst,etot_weight)
    CO.add_contribution(rmse_fattst,ftot_weight)
    CO.add_contribution(rmse_sattst,stot_weight)
    # commented examples on how to use energy differences in the objective function
    # a SINGLE structure is added to two new fitsnap groups, the corresponding energy
    # difference between group 1 and group 2 is given as the target (in eV) (NOT eV/atom)
    #obj_ads = ediff_cost(fittst,g1='H_ads_W_1',g2='H_above_W_1',target=-777.91314380000-(-775.10721929000),grouptype=grouptype)
    #CO.add_contribution(obj_ads,etot_weight * 1)
    #obj_ads = ediff_cost(fittst,g1='H_ads_W_2',g2='H_above_W_2',target=-781.72430764000-(-781.66087408000),grouptype=grouptype)
    #CO.add_contribution(obj_ads,etot_weight * 1)
    cost = CO.evaluate_cost()
    del CO
    return cost

# @snap.pt.rank_zero
def seed_maker(snap, mc,mmax = 1000000000,use_saved_seeds=True):
    if use_saved_seeds:
        try:
            seeds = np.load('seeds.npy')
            if np.shape(seeds)[0] < mc:
                snap.pt.single_print('potentially not enough seeds for this run, appending more')
                seeds = np.append(seeds , np.random.randint(0,mmax, mc- (np.shape(seeds)[0])  ))
                np.save('seeds.npy',seeds)
            else:
                seeds = seeds
        except FileNotFoundError:
            seeds = np.random.randint(0,mmax,mc)
            np.save('seeds.npy',seeds)
    else:
        seeds = np.random.randint(0,mmax,mc)
        np.save('seeds.npy',seeds)
    return seeds

# @snap.pt.rank_zero
def mutation(current_w_combo, current_ef_rat, current_es_rat, my_w_ranges,my_ef_ratios, my_es_ratios,ng, w_combo_delta=np.array([]), ef_rat_delta=np.array([]), s_combo_delta=np.array([]), apply_random=True, full_mutation=False):
    if type(current_w_combo) == tuple:
        current_w_combo = np.array(current_w_combo)
    if type(current_ef_rat) == tuple:
        current_ef_rat = np.array(current_ef_rat)
    if type(current_es_rat) == tuple:
        current_es_rat = np.array(current_es_rat)
    if full_mutation:
        if apply_random:
            test_w_combo = np.random.rand()*np.random.choice(my_w_ranges,ng)
            test_ef_rat = np.random.rand()*np.random.choice(my_ef_ratios,ng)
            test_es_rat = np.random.rand()*np.random.choice(my_es_ratios,ng)
        else:
            test_w_combo = np.random.choice(my_w_ranges,ng)
            test_ef_rat = np.random.choice(my_ef_ratios,ng)
            test_es_rat = np.random.choice(my_es_ratios,ng)
    else:
        test_w_combo = current_w_combo.copy()
        test_ef_rat = current_ef_rat.copy()
        test_es_rat = current_es_rat.copy()
        test_w_ind = np.random.choice(range(ng))
        if apply_random:
            plusvsprd =  1 #TODO implement addition/product steps after constraining min/max weights
            if plusvsprd:
                test_w_combo[test_w_ind] = np.random.rand() * np.random.choice(my_w_ranges)
                test_ef_rat[test_w_ind] = np.random.rand() * np.random.choice(my_ef_ratios)
                test_es_rat[test_w_ind] = np.random.rand() * np.random.choice(my_es_ratios)
            else:
                test_w_combo[test_w_ind] *= np.random.rand() * np.random.choice(my_w_ranges)
                test_ef_rat[test_w_ind] *= np.random.rand() * np.random.choice(my_ef_ratios)
                test_es_rat[test_w_ind] *= np.random.rand() * np.random.choice(my_es_ratios)
        else:
            test_w_combo[test_w_ind] = np.random.choice(my_w_ranges)
            test_ef_rat[test_w_ind] = np.random.choice(my_ef_ratios)
            test_es_rat[test_w_ind] = np.random.choice(my_es_ratios)
    if np.shape(w_combo_delta)[0] != 0:
        return test_w_combo * w_combo_delta, test_ef_rat*ef_rat_delta, test_es_rat*s_combo_delta
    else:
        return test_w_combo,test_ef_rat,test_es_rat


# @snap.pt.rank_zero
def print_final(snap, gtks, ew_frcrat_final, write_to_json=False):
    ew_final, frcrat_final, srcrat_final = ew_frcrat_final

    calc_stress = snap.config.sections["CALCULATOR"].stress
    print_stress = True
    wcols = [v for v in snap.config.sections["GROUPS"].group_sections if "weight" in v]
    num_wcols = len(wcols)
    if num_wcols == 2:
        print_stress = False
        
    # fitsnap TODO: to accurately output the best weights, we also need the train/test split specified by the user. at least when using the JSON scraper, training_size and_testing size are converted from floats into integers, which is inconsistent and should be updated. to work around that for now, we take the testing_size and training_size integers and convert them back into fractions. these will probably be very similar to the user's input, but may vary a little bit
    loc_gt = snap.config.sections["GROUPS"].group_table

    collect_lines = []
    snap.pt.single_print('\n--> Best group weights:')
    for idi, dat in enumerate(gtks):
        en_weight = ew_final[idi]
        frc_weight = ew_final[idi]*frcrat_final[idi]
        if print_stress:
            src_weight = ew_final[idi]*srcrat_final[idi]

        ntrain = loc_gt[dat]['training_size']
        ntest = loc_gt[dat]['testing_size']
        ntot = ntrain + ntest
        train_sz = round(ntrain/ntot,2)
        test_sz = round(ntest/ntot,2)

        # snap.pt.single_print('%s       =  %1.2f      %1.2f      %.16E      %.16E      1.E-12' % (dat, train_sz,test_sz,en_weight,frc_weight))
        group_line = f'{dat}       =  {train_sz}      {test_sz}      {en_weight}      {frc_weight}'
        if print_stress:
            group_line += f"      {src_weight}"
        snap.pt.single_print(group_line)
        collect_lines.append([dat, group_line.replace(f'{dat}       =  ','')])
    snap.pt.single_print("")

    # MEG NOTE: this write_to_json works fine but is a bit of a mess
    if write_to_json:
        infile_name = snap.config.infile
        settings = snap.config.indict

        # if a dictionary wasn't used to import settings, create one from the input file
        # TODO can probably refactor in more FitSNAP API-friendly way... would also avoid case-sensitivity bug
        if settings == None:
            # read in config again
            import configparser
            c = configparser.ConfigParser()
            c.read(infile_name)
            settings = {s:dict(c.items(s)) for s in c.sections()}

        # remove stress parameters and update smartweights from config object
        # TODO make sure to manage this if stress/smartweights management changes
        settings["GROUPS"]["group_sections"] = " ".join([gs for gs in settings["GROUPS"]["group_sections"].split() if gs != "vweight"])
        settings["GROUPS"]["group_types"] = " ".join([gt for gt in settings["GROUPS"]["group_types"].split()][:-1])
        settings["GROUPS"]["smartweights"] = str(snap.config.sections["GROUPS"].smartweights)

        # automatically create an outfile name from the potential name
        # TODO eventually allow user to override
        potential = settings["OUTFILE"]["potential"]
        outfile = f"{potential}.fs-input.json"

        # update group weights string in settings
        for group, line in collect_lines:
            settings["GROUPS"][group] = line

        # currently a bug where some lower/uppercase versions of groups are doubled
        del_invalid = []
        lower_gtks = [g.lower() for g in gtks]
        for sgroup in settings["GROUPS"].keys():
            input_path = snap.config.sections["PATH"].datapath
            group_path = f"{input_path}/{sgroup}"
            if not os.path.exists(group_path) and sgroup in lower_gtks:
                del_invalid.append(sgroup)
        [settings["GROUPS"].pop(g) for g in del_invalid] 

        # write to json
        snap.pt.single_print("Writing to JSON: ", outfile)
        with open(outfile, 'w') as f:
            json.dump(settings, f, indent=4)

#LOGAN NOTE: HAVE NOT UPDATED THIS FUNCTION - DON'T THINK IT NEEDS IT?
# MEG NOTE: i think you're correct but let's have James take a look
def latin_hypercube_sample(variable_ranges_dict, variable_types_dict, num_samples, seed=12345):
    # TODO is this doubled from lhparams, should be removed?
    # TODO if not, should this be varied or taken from lhparams?
    np.random.seed(seed)
    variable_ranges = [tuple(v) for v in list(variable_ranges_dict.values())]
    variable_types = [typ for typ in list(variable_types_dict.values())]
    num_variables = len(variable_ranges)
    varseeds = np.random.randint(0,10000,num_variables)
    samples_per_variable = num_samples // num_variables
    #samples_per_variable = 1
    # print (samples_per_variable,num_samples)
    # Generate the initial Latin Hypercube
    lhs_matrix = np.zeros((num_samples, num_variables))
    #(24, 2) (26, 24) nsamp 26
    # print ('lh info',np.shape(variable_ranges),np.shape(lhs_matrix),'nsamp',num_samples)
    for i in range(num_variables):
        np.random.seed(varseeds[i])
        vtyp = variable_types[i]
        # print (i,'vtyp',vtyp,'varranges',variable_ranges[i])#,(min(variable_ranges),max(variable_ranges) ))
        if vtyp == float:
            lhs_matrix[:, i] = np.random.uniform(min(variable_ranges[i]), max(variable_ranges[i]), num_samples)
        elif vtyp == int:
            vrange = list(range(min(variable_ranges[i]),max(variable_ranges[i]) +1,1))
            lhs_matrix[:, i] = np.random.choice(vrange,size=num_samples)
        elif vtyp == 'logfloat':
            #vrange = list(range(min(variable_ranges[i]),max(variable_ranges[i]) +1,1))
            coldata = np.array([10**kk for kk in np.random.uniform(min(variable_ranges[i]), max(variable_ranges[i]), num_samples)])
            # print ('col info',i,coldata)
            lhs_matrix[:, i] = coldata #np.array([10**kk for kk in np.random.choice(vrange,size=num_samples)])
        elif vtyp == str:
            lhs_matrix[:, i] = np.random.choice(variable_ranges[i],size=num_samples)
        else:
            raise TypeError("variable type not implemented")

    # Shuffle the samples within each variable column
    for i in range(num_variables):
        np.random.shuffle(lhs_matrix[:, i])
    # Randomly select one sample from each column to form the final Latin Hypercube
    lhs_samples = np.zeros((num_samples, num_variables))
    for i in range(num_variables):
        lhs_samples[0 :num_samples , i] = lhs_matrix[0 :num_samples , i]

    return lhs_samples


def prep_fitsnap_input(snap, smartweights_override=False):
    # turn off smartweights and warn user
    smartweights = snap.config.sections["GROUPS"].smartweights
    if smartweights == 1 and not smartweights_override:
      snap.pt.single_print(f"\nWARNING: Smartweights toggled on, but is not recommended for current version of genetic_algorithm.")
      snap.pt.single_print(f"WARNING: Setting smartweights to 0.")
      snap.pt.single_print(f"WARNING: Use the argument 'smartweights_override=True' to keep smartweights on.\n")
      snap.config.sections["GROUPS"].smartweights = 0

    # if the number of groups is three or less, crossover operations will fail
    # in goffptimize_output, crossover function: cpt = np.random.randint(1, ne-2)
    # for now, elegantly crash if user has 3 or fewer groups 
    num_groups = len(snap.config.sections["GROUPS"].group_table.keys())
    if num_groups <= 3:
        snap.pt.single_print("\n")
        snap.pt.single_print("\n!ERROR: Need 4 or more groups to use genetic algorithm (see comment)!")
        snap.single_print("!ERROR: I am elegantly crashing now so that you can contact the FitSNAP team to have them solve this for you!!")
        snap.pt.single_print("\n")
        exit()

    # turn off fitting to stresses
    # LOGAN NOTE: TODO: HANDLE NO STRESS FITTING WITH SMART REMOVAL OF STRESS VARIABLES IN THIS CODE 
    # MEG NOTE: we could put the removal here, but i think it makes more sense within the GA itself using the snap.config object settings. 
    # this function serves to 1) check and make minor changes to the FitSNAP config, and 2) warn the user about that. 
    # maybe we need to TODO factor out this function and just put those warnings right where things are changed? 
    # or is it better to warn before the initial fit, to give the panicked user enough time to hit CTRL+C a billion times before it starts rolling?
    calc_stress = snap.config.sections["CALCULATOR"].stress
    has_vweights = True if "vweight" in snap.config.sections["GROUPS"].group_sections else False
    if not calc_stress and has_vweights:
        snap.pt.single_print("\n")
        snap.pt.single_print(f"!WARNING: Your FitSNAP input script indicates you don't want to fit to stresses ([CALCULATOR] stress = 0), but your [GROUPS] have a 'vweights' column!")
        snap.pt.single_print(f"!WARNING: Since you don't want to fit to stresses, we're gonna populate all vweights columns with 0!")
        snap.pt.single_print(f"!WARNING: We're just warning you here because the output might be confusing.")
        snap.pt.single_print(f"!WARNING: <---- consider yourself warned!")
        snap.pt.single_print("\n")
    if calc_stress and not has_vweights:
        snap.pt.single_print("\n")
        snap.pt.single_print(f"!ERROR: Your FitSNAP input script indicates you want to fit to stresses ([CALCULATOR] stress = 1), but your [GROUPS] section is MISSING the 'vweights' column!")
        snap.pt.single_print(f"!ERROR: To fix this, add the word 'vweights' to the end of the [GROUPS] group_section variable, 'float' to the end of the [GROUPS] group_types variable, and some number (doesn't matter what) to the end of each of your group's weights.")
        snap.pt.single_print(f"!ERROR: Try again after adding that stuff! Now exiting.")
        snap.pt.single_print("\n")
        exit()

#-----------------------------------------------------------------------
# begin the primary optimzation functions
#-----------------------------------------------------------------------

# @snap.pt.rank_zero
def sim_anneal(snap):  ##LOGAN NOTE: I have not yet updated this function
    #---------------------------------------------------------------------------
    # Begin optimization hyperparameters
    time1 = time.time()

    # get groups and weights 
    gtks = snap.config.sections["GROUPS"].group_table.keys()
    gtks = list(gtks)
    snap.pt.single_print('Groups:', gtks)
    snap.pt.single_print('\n')

    # check if fitting to stresses turned on
    # if not, then set all stress weights to zero by populating stress_delta_keywords with all group names
    # a warning about this behavior is included in this module's "prep_fitnsap_input" function
    calc_stress = snap.config.sections["CALCULATOR"].stress
    if calc_stress:
        snap.pt.single_print("Stress fitting not yet implemented for simulated anneal!")
        snap.pt.all_barrier()
        return 0
    #for future implementation
    #if not calc_stress:
        #stress_delta_keywords = gtks

    
    size_b = np.shape(snap.pt.fitsnap_dict['Row_Type'])[0]
    grouptype = snap.pt.fitsnap_dict['Groups'].copy()
    rowtype = snap.pt.fitsnap_dict['Row_Type'].copy()

    etot_weight = 1.0
    ftot_weight = 1.5
    rmse_tot = 500

    # sampling magnitudes per hyperparameter
    my_w_ranges = [1.e-3,1.e-2,1.e-1,1.e0,1.e1,1.e2,1.e3]
    my_ef_ratios = [0.1,1,10]

    # Artificial temperatures
    betas = [1.e0,1.e1,1.e2,1.e3,1.e4]
    # Max number of steps per artificial temperature
    count_per_beta = [400,400,600,1000,1000]
    # threshhold for convergence of cost function
    thresh = 0.005

    seedpad = 50
    #build seeds (uses saved seeds by default)
    countmaxtot = int(np.sum(count_per_beta))
    seedsi = seed_maker(snap, countmaxtot + seedpad)

    # End optimization hyperparameters
    #---------------------------------------------------------------------------
    
    tot_count = 0
    #threshold for cost function before accepting model
    current_w_combo = [1.e0]*len(gtks)
    current_ef_rat = [10]*len(gtks)
    tot_count = 0
    apply_random = True # flag to select a single random hyperparam to step rather than stepping all hyperparams
    naccept = 0
    np.random.seed(seedsi[tot_count])
    # loop over fictitious temperatures
    for ibeta,beta in enumerate(betas):
        count = 0
        naccepti = 0
        maxcount = count_per_beta[ibeta]
        # propose trial weights while counts are below maximums and 
        # objective function is above threshhold
        while count <= maxcount and rmse_tot >= thresh:

            if tot_count <= 5: # allow for large steps early in simulation
                test_w_combo, test_ef_rat = mutation(current_w_combo,current_ef_rat,my_w_ranges,my_ef_ratios,ng=len(gtks),apply_random=True,full_mutation=True)
            else:
                test_w_combo, test_ef_rat = mutation(current_w_combo,current_ef_rat,my_w_ranges,my_ef_ratios,ng=len(gtks),apply_random=True,full_mutation=False)

            update_weights(snap, test_w_combo, test_ef_rat, gtks, size_b, grouptype)

            rmse_tottst = fit_and_cost(snap,[etot_weight,ftot_weight])

            delta_Q = rmse_tottst - rmse_tot
            boltz = np.exp(-beta*delta_Q)
            rndi = np.random.rand()
            logical = rndi <= boltz
            if logical:
                naccept += 1
                naccepti +=1
                rmse_tot = rmse_tottst
                current_w_combo = test_w_combo
                current_ef_rat = test_ef_rat

            meta = (tuple(list(current_w_combo)),) + (tuple(list(current_ef_rat)),)
            count += 1
            snap.pt.single_print('beta',beta,'count',count,' accept ratio for current beta %f' % (naccepti/count) ,meta,boltz,rmse_tottst,rmse_tot)
            tot_count += 1
            np.random.seed(seedsi[tot_count])

    # write output for optimized potential
    print_final(meta)
    time2 = time.time()
    snap.pt.single_print('Total optimization time,', time2 - time1, 'total number of fits', tot_count)
    snap.write_output()


# @snap.pt.rank_zero
def genetic_algorithm(snap, population_size=50, ngenerations=100, my_w_ranges=[1.e-4,1.e-3,1.e-2,1.e-1,1,1.e1,1.e2,1.e3,1.e4], my_ef_ratios=[0.001,0.01,0.1,1,10,100,1000], etot_weight=1.0, ftot_weight=1.0, stot_weight=1.0, r_cross=0.9, r_mut=0.1, conv_thr = 1.E-10, conv_check = 2., force_delta_keywords=[], stress_delta_keywords=[], write_to_json=False, my_es_ratios=[], use_initial_weights_flag=False ):
    #---------------------------------------------------------------------------
    # Begin in-function optimization hyperparameters
    # snap: FitSnap instance being handled by genetic algorithm
    # population_size: number of candidates ("creatures") generated within one generation and tested for fitness. in this code, fitness is how well group weights perform in a FitSnap fit (no puns intended)
    # ngenerations: maximum number of allowed iterations of populations. this ends the genetic algorithm calculations if the convergence threshold (conv_thr, see below) is not reached beforehand
    # my_w_ranges: allowed scaling factors for energy weights
    # my_ef_ratios: allowed scaling factors for force weights
    # etot_weight and ftot_weight: weights for energy and force rmse in the optimizer cost function
    # r_cross and r_mut: cross over (parenting) and mutation hyperparameters
    # conv_thr: convergence threshold for full function (value of RMSE E + RMSE F at which simulation is terminated" 
    # conv_check: fraction of ngenerations to start checking for convergence (convergence checks wont be performed very early)
    time1 = time.time()

    # get all group names  
    gtks = snap.config.sections["GROUPS"].group_table.keys()
    gtks = list(gtks)
    snap.pt.single_print('Groups:', gtks)
    snap.pt.single_print('\n')

    # check if fitting to stresses turned on
    # if not, then set all stress weights to zero by populating stress_delta_keywords with all group names
    # a warning about this behavior is included in this module's "prep_fitnsap_input" function
    calc_stress = snap.config.sections["CALCULATOR"].stress
    if not calc_stress:
        stress_delta_keywords = gtks
    
    # all calculations must include 'vweights' column for internal calculations
    # MEG NOTE: we could refactor all internal GA inputs into arrays with 2 or 3 cols to get around this, but that's a bigger overhaul and not really important for now
    wcols = [v for v in snap.config.sections["GROUPS"].group_sections if "weight" in v]
    num_wcols = len(wcols)
    if num_wcols == 2:
        for key in gtks:
            snap.config.sections["GROUPS"].group_table[key]['vweight'] = 0.0

    # population of generations
    # population can't have odd numbers currently
    if population_size % 2 == 1:
        snap.pt.single_print("\n")
        snap.pt.single_print(f"WARNING: Cannot use odd numbers for population size (input: {population_size})")
        snap.pt.single_print(f"WARNING: Updating population_size: population_size +=1 (larger populations seem to perform better, in other GAs at least)")
        snap.pt.single_print(f"WARNING: New population_size: {population_size+1})")
        population_size += 1
        snap.pt.single_print("\n")

    # start getting weights 
    initial_weights={}
    if use_initial_weights_flag:
        for key in gtks:
            initial_weights[key] = [snap.config.sections["GROUPS"].group_table[key]['eweight'], snap.config.sections["GROUPS"].group_table[key]['fweight'], \
                                    snap.config.sections["GROUPS"].group_table[key]['vweight']]

    size_b = np.shape(snap.pt.fitsnap_dict['Row_Type'])[0]
    grouptype = snap.pt.fitsnap_dict['Groups'].copy()
    rowtype = snap.pt.fitsnap_dict['Row_Type'].copy()

    countmaxtot = int(population_size*(ngenerations+2))
    seedsi = seed_maker(snap, countmaxtot)

    #number of hyperparameters:
    # num of energy group weights
    ne = len(gtks)
    # num of force group weights
    nf = ne
    ns = ne
    # total
    if not my_es_ratios:
        nh = ne + nf
    else:
        nh = ne + nf + ns

    # update ranges and ratios
    eranges = [my_w_ranges]
    ffactors = [my_ef_ratios]
    sfactors = [my_es_ratios]  ##LOGAN NOTE: EMPTY LIST IF NOT USING

    # selection method (only tournament is currently implemented)
    # TODO implement other methods?
    selection_method = 'tournament'

    # modify convergence check for new conv_flag
    check_gen = int((ngenerations/conv_check))

    # End optimization hyperparameters
    #---------------------------------------------------------------------------

    # set up generation 0
    best_eval = 9999999.9999
    conv_flag = False
    first_seeds = seedsi[:population_size+1]
    hp = HyperparameterStruct(ne,nf,ns,eranges,ffactors,sfactors)

    # population = [hp.random_params(inputseed=first_seeds[ip]) for ip in range(population_size)] # TODO orig

    # Random initial population for first generation:
    #population = [hp.random_params(inputseed=first_seeds[ip]) for ip in range(population_size)]
    # Latin hypercube population for first generation:
    population = hp.lhs_params(num_samples=population_size,inputseed=first_seeds[0])

    generation = 0

    best_evals = [best_eval]
    best = tuple(population[0])
    sim_seeds = seedsi[population_size:]
    np.random.seed(sim_seeds[generation])
    w_combo_delta = np.ones(len(gtks))

    # delta function to zero out force weights on structures without forces  
    # now implemented with user-specified keywords
    if force_delta_keywords != []:
        not_in_fdkws = lambda gti: all([True if fdkw not in gti else False for fdkw in force_delta_keywords])
        ef_rat_delta = np.array([1.0 if not_in_fdkws(gti) else 0.0 for gti in gtks])
    else:
        ef_rat_delta = np.array([1.0]*len(gtks))

    if stress_delta_keywords != [] or len(stress_delta_keywords) == len(gtks):
        not_in_fdkws = lambda gti: all([True if fdkw not in gti else False for fdkw in stress_delta_keywords])
        es_rat_delta = np.array([1.0 if not_in_fdkws(gti) else 0.0 for gti in gtks])
    else:
        es_rat_delta = np.array([1.0]*len(gtks))
        
    while generation <= ngenerations and best_eval > conv_thr and not conv_flag:
        scores = []
        # current generation
        for creature in population:
            creature_ew, creature_ffac, creature_sfac = tuple(creature.reshape(3,ne).tolist())
            creature_ew = tuple(creature_ew)
            creature_ffac = tuple(creature_ffac)
            creature_sfac = tuple(creature_sfac)

            ##LOGAN NOTE: should confirm this is working as expected (always multiplying factor by initial weight and not a previous generation product by initial weight)
            update_weights(snap, creature_ew, creature_ffac, creature_sfac, gtks, size_b, grouptype,initial_weights=initial_weights)
            costi = fit_and_cost(snap,[etot_weight,ftot_weight,stot_weight])
            scores.append(costi)

            #NOTE to add another contribution to the cost function , you need to evaluate it in the loop
            # and add it to the fit_and_cost function
            # if this involves a lammps simulation, you will have to print potentials at the different steps
            # to run the lammps/pylammps simulation. To do so, the fitsnap output file name prefix should
            # be updated per step, then snap.write_output() should be called per step. This will likely increase
            # the optimization time.
        # Anything printed with snap.pt.single_print will be included in output file.
        snap.pt.single_print('Generation, scores, popsize:',generation,len(scores),population_size)

        # Print generation and best fit.
        # bestfit = min(scores)
        # print(f"{generation} {bestfit}")
        for i in range(population_size):
            if scores[i] < best_eval:
                best, best_eval = tuple(population[i]), scores[i]
        best_evals.append(best_eval)
        try:
            ## Original flag
            conv_flag = np.round(np.var(best_evals[int(ngenerations/conv_check)-int(ngenerations/10):]),14) == 0
            ## New flag, currently testing
            # conv_flag = np.round(np.var(best_evals[-(check_gen*math.floor(len(best_evals)/check_gen)):]),14) == 0
        except IndexError:
            conv_flag = False
        printbest = tuple([tuple(ijk) for ijk in np.array(best).reshape(3,ne).tolist()])
        snap.pt.single_print('\n')
        snap.pt.single_print('Generation:',generation, 'score:', scores[i])

        # TODO input user's settings or warn user that it will be overwritten (train_sz, test_sz)
        print_final(snap, gtks, printbest)
        slct = Selector(selection_style = selection_method)
        selected = [slct.selection(population, scores) for creature_idx in range(population_size)]
        del slct

        # new generation
        children = list()
        for ii in range(0, population_size, 2):
            # get selected parents in pairs
            p1, p2 = selected[ii], selected[ii+1]
            # crossover and mutation
            rndcross, rndmut = tuple(np.random.rand(2).tolist())
            if rndcross <= r_cross:
                cs = crossover(p1, p2, len(gtks), w_combo_delta, ef_rat_delta, es_rat_delta)
            else:
                cs = [p1,p2]
            for c in cs:
                # mutation
                if rndmut <= r_mut:
                    current_creature_ew, current_creature_ffac, current_creature_sfac = tuple(c.reshape(3,ne))
                    current_creature_ew = tuple(current_creature_ew)
                    current_creature_ffac = tuple(current_creature_ffac)
                    current_creature_sfac = tuple(current_creature_sfac)

                    mutated_creature_ew, mutated_creature_ffac, mutated_creature_sfac = mutation(current_creature_ew,current_creature_ffac,current_creature_sfac,\
                                                                                                 my_w_ranges,my_ef_ratios,my_es_ratios,ng=len(gtks),\
                                                                                                 w_combo_delta=w_combo_delta,ef_rat_delta=ef_rat_delta, s_combo_delta=es_rat_delta,\
                    apply_random=True,
                    full_mutation=False)

                    c = np.concatenate((mutated_creature_ew,mutated_creature_ffac,mutated_creature_sfac))
                    # store for next generation
                children.append(c)
        generation += 1
        np.random.seed(sim_seeds[generation])
        population = children
    best_ew, best_ffac, best_sfac = tuple(np.array(best).reshape(3,ne).tolist())
    best_ew = tuple(creature_ew)
    best_ffac = tuple(creature_ffac)
    best_sfac = tuple(creature_sfac)

    ##LOGAN NOTE: should confirm this is working as expected (always multiplying factor by initial weight and not a previous generation product by initial weight)
    update_weights(snap, best_ew, best_ffac, best_sfac, gtks, size_b, grouptype, initial_weights=initial_weights)
    costi = fit_and_cost(snap,[etot_weight,ftot_weight,stot_weight])
    print_final(snap, gtks, tuple([best_ew,best_ffac,best_sfac]), write_to_json=write_to_json)
    time2 = time.time()
    elapsed = round(time2 - time1, 2)
    snap.pt.single_print(f'Total optimization time: {elapsed} s')
    snap.pt.single_print('Writing final output')
    snap.write_output()