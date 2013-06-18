#! /usr/bin/env python


import os
from copy import deepcopy
from numpy import array,empty

from generic import obj
from project import generate_physical_system
from project import generate_pwscf
from project import generate_pw2casino,generate_pw2qmcpack
from project import generate_sqd
from project import generate_qmcpack,generate_jastrow
from project import loop,vmc,dmc,linear,cslinear
from project import settings,ProjectManager,Job,Pobj

from qmcpack_calculations import basic_qmc,standard_qmc

from periodic_table import periodic_table
from unit_converter import convert
from developer import DevBase
from plotting import *
from debug import *




def sort_pseudos(pps,dext=set(['ncpp','upf']),qext=set(['xml'])):
    dpp = []
    qpp = []
    for pp in pps:
        n,ext = pp.lower().rsplit('.',1)
        if ext in dext:
            dpp.append(pp)
        elif ext in qext:
            qpp.append(pp)
        else:
            print 'Error: pseudopotential format could not be determined\n  filename: {0}\n  extensions checked: {1}'.format(pp,list(dext)+list(qext))
            exit()
        #end if
    #end for
    return dpp,qpp
#end def sort_pseudos




class ColorWheel(DevBase):
    def __init__(self):
        colors = 'Black Maroon DarkOrange Green DarkBlue Purple Gray Firebrick Orange MediumSeaGreen DodgerBlue MediumOrchid'.split()
        styles = '- -- -. :'.split()
        c = []
        for style in styles:
            for color in colors:
                c.append((color,style))
            #end for
        #end for
        self.colors = c
        self.icolor = -1
    #end def __init__


    def next(self):
        self.icolor = (self.icolor+1)%len(self.colors)
        return self.colors[self.icolor]
    #end def next


    def reset(self):
        self.icolor = -1
    #end def reset
#end class ColorWheel
color_wheel = ColorWheel()
    



class ValidatePPBase(DevBase):
    settings = None

    def performed_runs(self):
        s = self.settings
        return not s.generate_only and not s.status_only
    #end def performed_runs
#end class ValidatePPBase




class ValidationStage(ValidatePPBase):
    stage_inputs = set()
    stage_dependencies = set()
    stage_result = None
    final_result = None


    stage_type = None
    data_type  = None
    title      = None
    xlabel     = None
    ylabel     = None
    xunit      = None
    yunit      = None


    def __init__(self):
        self.set(
            name         = None,
            ready        = False,
            sims         = obj(),
            saved_inputs = obj(),
            simlist      = [],
            inputs       = obj(),
            results      = obj(),
            result_sims  = None
            )
    #end def __init__


    def get_tags(self):
        s=self
        return s.title,s.xlabel,s.ylabel,s.xunit,s.yunit,s.stage_type,s.data_type
    #end def get_tags


    def set_name(self,name):
        self.name = name
    #end def set_name


    def set_result_sims(self,result_sims):
        self.result_sims = result_sims
    #end def set_result_sims    


    def check_dependencies(self):
        inputs = self.inputs
        results = self.results
        have_inputs = True
        for inp in self.stage_inputs:
            have_inputs = have_inputs and inp in inputs and inputs[inp]!=None
        #end for
        have_deps = True
        for dep in self.stage_dependencies:
            have_deps = have_deps and dep in results and results[dep]!=None
        #end for
        self.ready = have_inputs and have_deps
    #end def check_dependencies


    def set_stage_inputs(self,**inputs):
        self.inputs.set(**inputs)
    #end def set_stage_inputs


    def set_stage_results(self,**results):
        self.results.set(**results)
    #end def set_stage_results


    def check_inputs(self,inputs):
        inkeys = set(inputs.keys())
        if not self.stage_inputs < inkeys:
            self.error('stage inputs are missing\n  inputs required: {0}\n  inputs provided: {1}\n  key: {2}'.format(list(self.stage_inputs),list(inkeys)),key)
        #end if
        if not self.stage_dependencies < inkeys:
            self.error('stage dependencies are missing\n  dependencies required: {0}\n  dependencies provided: {1}\n  key: {2}'.format(list(self.stage_dependencies),list(inkeys)),key)
        #end if
    #end def check_inputs

        
    def save_sims(self,sims,key,inputs,inputs_save=None):
        if inputs_save is None:
            inputs_save = inputs
        #end if
        if 'result' in sims:
            if self.stage_result is None:
                self.error('encountered result sim for a stage with no results')
            #end if
            self.result_sims[self.stage_result,key] = sims.result
            del sims.result
        #end if
        self.sims[key] = sims
        self.saved_inputs[key] = inputs_save
        block = False
        for dep in self.stage_dependencies:
            if inputs[dep] is None:
                block = True
            #end if
        #end for
        if not block:
            self.simlist.extend(sims.list())
        #end if
    #end def save_sims

            
    def get_result_sim(self,key):
        if not key in self.result_sims:
            self.error('attempted to retrieve non-existent stage result\n  key requested: {0}\n  keys present: {1}'.format(key,self.result_sims.keys()))
        #end if
        return self.result_sims[key]
    #end def get_result_sim


    def make_all_sims(self,basepath):
        self.not_implemented()
    #end def make_all_sims


    def sim_list(self):
        return self.simlist
    #end def sim_list


    def get_results(self):
        self.not_implemented()
    #end def get_results


    def status(self,pad=' ',n=0,*args):
        if self.ready:
            status = 'ready'
        else:
            status = 'waiting'
        #end if
        print n*pad+'{0:<20}  {1}'.format(self.name,status)
        if not self.ready or 'verbose' in args:
            p = (n+1)*pad
            presence = {True:'present',False:'absent'}
            for name in self.stage_inputs:
                print p+'{0:<20}  {1}'.format(name,presence[name in self.inputs])
            #end for
            for name in self.stage_dependencies:
                print p+'{0:<20}  {1}'.format(name,presence[name in self.results])
            #end for
        #end if        
    #end def status
#end class ValidationStage




class ValidationProcess(ValidatePPBase):

    allowed_stage_inputs  = set()
    allowed_stage_results = set()

    def __init__(self,**kwargs):
        self.name = None
        self.stages = obj()
        self.stage_order = None
        self.results = obj()
        si = obj()
        for name in self.allowed_stage_inputs:
            si[name] = None
        #end for
        self.stage_inputs = si
        sr = obj()
        for name in self.allowed_stage_results:
            sr[name] = None
        #end for
        self.stage_results = sr
        self.result_sims = obj()
        self.initialize(**kwargs)
        for stage in self.stages:
            stage.set_result_sims(self.result_sims)
        #end for    
    #end def __init__

        
    def initialize(self,**kwargs):
        self.not_implemented()
    #end def initialize


    def set_name(self,name):
        self.name = name
    #end def set_name

        
    def get_stage_order(self):
        if self.stage_order!=None:
            stage_order = self.stage_order
        else:
            stage_order = self.stages.keys()
            stage_order.sort()
        #end if
        return stage_order
    #end def get_stage_order


    def status(self,pad='  ',n=0,*args):
        print n*pad+self.name
        stage_order = self.get_stage_order()
        for sname in stage_order:
            self.stages[sname].status(pad,n+1,*args)
        #end for
        print n*pad+'end '+self.name
    #end def status


    known_functionals = set('lda pbe pw91 hse pbe0'.split())
    pp_extensions = [['upf','ncpp'],['xml']]

    def check_funcs_pseudos(self,**func_pp):
        ppmap = obj()
        ppdir = self.settings.pseudo_dir
        if not os.path.exists(ppdir):
            self.error('pseudopotential directory does not exist\n  directory provided: '+ppdir)
        #end if
        ppfiles = os.listdir(ppdir)
        ppfiles.sort()
        unknown_functionals = []
        missing_pp_files = []
        for functional,pseudos in func_pp.iteritems():
            if not functional in self.known_functionals:
                unknown_functionals.append(functional)
            else:
                for pseudo_prefix in pseudos:
                    pp_found  = []
                    for pp_ext in self.pp_extensions:
                        pp_prefix = pseudo_prefix.lower()
                        for ppfile in ppfiles:
                            prefix,ext = ppfile.rsplit('.',1)
                            prefix = prefix.lower()
                            ext = ext.lower()
                            if prefix==pp_prefix and ext in pp_ext:
                                pp_found.append(ppfile)
                                break
                            #end if
                        #end for
                    #end for
                    if len(pp_found)!=len(self.pp_extensions):
                        missing_pp_files.append(pseudo_prefix)
                    else:
                        ppmap[pseudo_prefix] = pp_found
                    #end if
                #end for
            #end if
        #end for
        msg=''
        if len(unknown_functionals)>0:
            msg+='inputted functionals are unrecognized\n  functionals: '+str(unknown_functionals)+'\n'
        #end if
        if len(missing_pp_files)>0:
            msg+='pseudopotential prefixes had no corresponding files\n  prefixes: '+str(missing_pp_files)+'\n  extensions required: '+str(self.pp_extensions)+'\n  directory searched: '+ppdir+'\n  directory contents: '+str(ppfiles)+'\n'
        #end if
        if msg!='':
            self.error(msg)
        #end if
        self.functionals = list(func_pp.keys())
        self.func_pp     = obj(**func_pp)
        self.ppmap       = ppmap
    #end def check_funcs_pseudos


    def pseudo_list(self):
        return list(self.ppmap.keys())
    #end def pseudo_list


    def add_stage(self,name,stage):
        if not isinstance(stage,ValidationStage):
            self.error(name+' is not a ValidationStage')
        #end if
        self.stages[name] = stage
        stage.set_name(name)
    #end def add_stage


    def add_stages(self,**stages):
        for name,stage in stages.iteritems():
            self.add_stage(name,stage)
        #end for
    #end def add_stages


    def set_stage_order(self,stage_order):
        stages = set(self.stages.keys())
        order = set(stage_order)
        if stages!=order:
            self.error('stage order must account for all stages\n  stages: {0}  stage_order: {1}'.format(list(stages),stage_order))
        #end if
        self.stage_order = stage_order
    #end def set_stage_order


    def set_stage_inputs(self,**inputs):
        inkeys = set(inputs.keys())
        invalid = inkeys-self.allowed_stage_inputs
        if len(invalid)>0:
            self.error('invalid stage inputs encountered\n  invalid inputs: {0}\n valid options are: {1}'.format(invalid,self.allowed_stage_inputs))
        #end if
        self.stage_inputs.set(**inputs)
        for stage in self.stages:
            stage.set_stage_inputs(**inputs)
        #end for
    #end def set_stage_inputs


    def check_stage_results(self,inkeys):
        invalid = set(inkeys)-self.allowed_stage_results
        if len(invalid)>0:
            self.error('invalid stage results encountered\n  invalid results: {0}\n valid options are: {1}'.format(invalid,self.allowed_stage_results))
        #end if
    #end def check_stage_results


    def set_stage_results(self,**results):
        self.stage_results.set(**results)
        for stage in self.stages:
            stage.set_stage_results(**results)
        #end for
    #end def set_stage_results


    def check_dependencies(self):
        for stage in self.stages:
            stage.check_dependencies()
        #end for
    #end def check_dependencies


    def pre_make_sims(self):
        None
    #end def pre_make_sims


    def make_sims(self,basepath):
        print 'make_sims',self.__class__.__name__
        self.pre_make_sims()
        self.check_dependencies()
        stage_order = self.get_stage_order()
        for name in stage_order:
            stage = self.stages[name]
            path = os.path.join(basepath,name)
            print ' ',name,stage.__class__.__name__,stage.ready
            if stage.ready:
                stage.make_all_sims(path)
            else:
                stage.status('  ',3,'verbose')
            #end if
        #end for
    #end def make_sims


    def sim_list(self):
        sims = []
        for stage in self.stages:
            sims.extend(stage.sim_list())
        #end for
        return sims
    #end def sim_list
#end class ValidationProcess



class ValidationProcesses(ValidatePPBase):

    process_type = None

    @classmethod
    def set_process_type(cls,ptype):
        cls.process_type = ptype
        cls.allowed_stage_inputs  = ptype.allowed_stage_inputs
        cls.allowed_stage_results = ptype.allowed_stage_results
    #end def set_process_type


    def __init__(self,**processes):
        self.name = None
        self.processes = obj()
        for pname,pinfo in processes.iteritems():
            pinfo['name'] = pname
            process = self.process_type(**pinfo)
            #process.set_name(pname)
            self.add_process(pname,process)
        #end for
        self.initialize()
    #end def __init__


    def set_name(self,name):
        self.name = name
    #end def set_name


    def status(self,pad='  ',n=0,*args):
        print n*pad+self.name
        pnames = self.processes.keys()
        pnames.sort()
        for pname in pnames:
            self.processes[pname].status(pad,n+1,*args)
        #end for
        print n*pad+'end '+self.name
    #end def status


    def initialize(self):
        None
    #end def initialize


    def add_process(self,name,process):
        if not isinstance(process,ValidationProcess):
            self.error('Validation input improperly constructed\n  {0} is not a ValidationProcess'.format(name))
        #end if
        self[name] = process
        self.processes[name] = process
    #end def add_process

        
    def sim_list(self):
        sims = []
        for process in self.processes:
            sims.extend(process.sim_list())
        #end for
        return sims
    #end def sim_list

    
    def pre_make_sims(self):
        None
    #end def pre_make_sims


    def make_sims(self,basepath):
        print 'make_sims',self.__class__.__name__
        self.pre_make_sims()
        for name,process in self.processes.iteritems():
            path = os.path.join(basepath,name)
            process.make_sims(path)
        #end for
    #end def make_sims


    def show_results(self,
                     stages     = None,
                     quantities = None,
                     processes  = None,
                     mode       = 'plot',
                     titles     = None,
                     xlabels    = None,
                     ylabels    = None,
                     xlims      = None,
                     ylims      = None,
                     lw         = 2,
                     units      = None,
                     xunits     = None,
                     figlevel   = 'quantity',
                     suppress   = False):
        if not self.performed_runs():
            return
        #end if
        modes = ['plot','print','all']
        figlevels = [None,'top','stage','quantity','process']    
        if not mode in modes:
            self.error('invalid mode encountered\n  you provided: {0}\n  valid options are: {1}'.format(mode,modes))
        #end if    
        if not figlevel in figlevels:
            self.error('invalid figlevel encountered\n  you provided: {0}\n  valid options are: {1}'.format(figlevel,figlevels))
        #end if    
        if stages is None:
            self.error('results cannot be shown unless the stages variable is specified')
        #end if    
        if isinstance(stages,str):
            stages = [stages]
        #end if    
        if quantities is None or isinstance(quantities,str):
            quantities = [quantities]
        #end if    
        if processes is None:
            processes = self.processes.keys()
        #end if
        plotting = mode=='plot'  or mode=='all'
        printing = mode=='print' or mode=='all'
        color_wheel.reset()
        if printing:
            print
            print '{0} results'.format(self.name)
        #end if
        if figlevel=='top' and plotting:
            figure()
        #end if    
        for sname in stages:
            if printing:
                print '  {0} stage results'.format(sname)
            #end if
            if figlevel=='stage' and plotting:
                figure()
            #end if    
            for qname in quantities:
                if qname!=None and printing:
                    print '    quantity {0} results'.format(qname)
                    pad = '      '
                else:
                    pad = '    '
                #end if
                if figlevel=='quantity' and plotting:
                    figure()
                #end if    
                for pname in processes:
                    if printing:
                        print pad+pname
                    #end if
                    if figlevel=='process' and plotting:
                        figure()
                    #end if    
                    if not pname in self.processes:
                        self.error('process {0} is unrecognized\n  valid options are: {1}'.format(pname,self.processes.keys()))
                    #end if
                    process = self.processes[pname]
                    if not sname in process.stages:
                        self.error('stage {0} is not part of process {1}\n  valid options are: {2}'.format(sname,process.name,process.stages.keys()))
                    #end if    
                    stage = process.stages[sname] 

                    #get simulation data from stage
                    #  simulation data as a function of keys
                    #  sort keys and cycle through colors, one per key
                    #  print or plot data for each key
                    #  stage should have default title info

                        
                    #  
                    #  title: x vs y
                    #               x (xu)    y (yu)  yerr (yu)
                    #  
                    #    k0
                    #      k1
                    #        k2     x         y       yerr
                    #               :         :        :


                    res = stage.get_results(quantity=qname)
                    rkeys = res.keys()
                    rkeys.sort()

                    titlet,xlabelt,ylabelt,xunitt,yunitt,stype,dtype = stage.get_tags()
                    if titles!=None:
                        ptitle = titles
                    else:
                        ptitle = titlet
                    #end if
                    if xlabels!=None:
                        pxlabel = xlabels
                    else:
                        pxlabel = xlabelt
                    #end if
                    if ylabels!=None:
                        pylabel = ylabels
                    else:
                        pylabel = ylabelt
                    #end if
                    

                    rtypes = {
                        ('value','scalar'):'y',
                        ('value','stat'  ):'ye',
                        ('scan' ,'scalar'):'xy',
                        ('scan' ,'stat'  ):'xye',
                        }
                    rtype = rtypes[stype,dtype]
                    convertx = xunits!=None and xunitt!=None
                    converty = units!=None and yunitt!=None
                    xunitb = ''
                    yunitb = ''
                    if convertx:
                        xunitb = '('+xunits+')'
                    elif xunitt!=None:
                        xunitb = '('+xunitt+')'
                    #end if
                    if converty:
                        yunitb = '('+units+')'
                    elif yunitt!=None:
                        yunitb = '('+yunitt+')'
                    #end if
                    if rtype=='y':
                        phead = '{0} {1:<4}'.format(ylabelt,yunitb)
                        pfmt  = '{0}     '
                        plotter = plot
                    elif rtype=='ye':
                        phead = '{0} {1:<4}  {2} {3:<4}'.format(ylabelt,yunitb,'error',yunitb)
                        pfmt  = '{0}       {1}'
                        plotter = errorbar
                    elif rtype=='xy':
                        phead = '{0} {1:<4}  {2} {3:<4}'.format(xlabelt,xunitb,ylabelt,yunitb)
                        pfmt  = '{0}       {1}'
                        plotter = plot
                    elif rtype=='xye':
                        phead = '{0} {1:<4}  {2} {3:<4}  {4} {5:<4}'.format(xlabelt,xunitb,ylabelt,yunitb,'error',yunitb)
                        pfmt  = '{0}       {1}       {2}'
                        plotter = errorbar
                    #end if
                    keylen = 0
                    for key in rkeys:
                        keylen = max(keylen,len(str(key)))
                    #end for
                    keylen+=4


                    if printing:
                        p = '  '
                        print pad+p+titlet
                        print pad+p+keylen*' '+phead
                    #end if
                    if stype=='value':
                        color,style = color_wheel.next()
                    #end if
                    for key in rkeys:
                        r = res[key]
                        x,y,yerr = None,None,None
                        xlen = 0
                        if stype=='value':
                            if isinstance(key,tuple):
                                x = key[-1] # hope this works
                            else:
                                x = key
                            #end if
                            if dtype=='scalar':
                                y    = r
                                yerr = 0
                            elif dtype=='stat':
                                y,yerr = r
                            #end if
                            x = array([x])
                            y = array([y])
                            yerr = array([yerr])
                            ufmt = pfmt
                        elif stype=='scan':
                            color,style = color_wheel.next()
                            if dtype=='scalar':
                                x,y = r
                                yerr = empty([])
                            elif dtype=='stat':
                                x,yp = r
                                yp = array(yp)
                                y    = yp[:,0]
                                yerr = yp[:,1]
                            #end if
                            x    = array(x)
                            y    = array(y)
                            yerr = array(yerr)
                            for xv in x:
                                xlen = max(xlen,len(str(xv)))
                            #end for
                            ufmt = pfmt.replace('{0}','{0:<'+str(xlen)+'}')
                        #end if
                        if convertx:
                            x = convert(x,xunitt,xunits)
                        #end if
                        if converty:
                            y    = convert(y,   yunitt,units)
                            yerr = convert(yerr,yunitt,units)
                        #end if
                        if rtype=='y':
                            data = array([y])
                            pdata = array([x,y])
                        elif rtype=='ye':
                            data = array([y,yerr])
                            pdata = array([x,y,yerr])
                        elif rtype=='xy':
                            data = array([x,y])
                            pdata = data
                        elif rtype=='xye':
                            data = array([x,y,yerr])
                            pdata = data
                        #end if

                        if printing:
                            skey = str(key)
                            if len(skey)<keylen:
                                skey+=(keylen-len(skey))*' '
                            #end if
                            skey = pad+2*p+skey
                            nw,nl = data.shape
                            for i in range(nl):
                                print skey+ufmt.format(*tuple(data[:,i]))
                                skey = pad+2*p+keylen*' '
                            #end for
                        #end if
                        if plotting:
                            plotter(*pdata,color=color,fmt=style,lw=lw,label=str(tuple(pname,*key)))
                            title(ptitle)
                            xlabel(pxlabel)
                            ylabel(pylabel)
                            xticks(x)
                        #end if
                        del x,y,yerr
                    #end for


                    if printing:
                        print pad+'end '+pname
                    #end if
                #end for
                if qname!=None and printing:
                    print '    quantity {0} results'.format(qname)
                #end if
            #end for
            if printing:
                print '  end {0} stage results'.format(sname)
            #end if
        #end for        
        if printing:
            print 'end {0} results'.format(self.name)
        #end if
        if mode=='plot' and not suppress:
            show()
        #end if    
        return
    #end def show_results

#end class ValidationProcesses




class AtomicValidationStage(ValidationStage):
    systypes = ['ae','pp']
    systype = None

    stage_type = 'scan'
    data_type  = 'stat'
    yunit      = 'Ha'


    def __init__(self,**vars):
        if not self.systype in self.systypes:
            self.error('system type {0} is unrecognized\n  valid system types are: {1}'.format(systype,self.systypes))
        #end if
        self.set(**vars)
        ValidationStage.__init__(self)
        svars = obj(
            stage_inputs = self.stage_inputs,
            stage_dependencies = self.stage_dependencies
            )
        prefix = self.systype+'_'
        for svar,varlist in svars.iteritems():
            prefixed = True
            for var in varlist:
                prefixed = prefixed and var.startswith(prefix)
            #end for
            if not prefixed:
                vars = []
                for var in varlist:
                    if var.startswith(prefix):
                        vars.append(var)
                    else:
                        vars.append(prefix+var)
                    #end if
                #end for
                self[svar] = vars
            #end if
        #end for
    #end def __init__


    def make_system(self,v):
        atom = self.system_info.atom
        if not 'q' in v:
            self.error('charge is not present, physical system cannot be made')
        #end if
        if self.systype=='ae':
            system = generate_physical_system(
                type       = 'atom',
                atom       =  atom,
                net_charge = v.q,
                net_spin   = v.q%2
                )
        elif self.systype=='pp':
            if not 'L' in v:
                self.error('box size is not present, physical system cannot be made')
            #end if
            L = v.L
            system = generate_physical_system(
                lattice    = 'orthorhombic',
                cell       = 'primitive',
                centering  = 'P',
                constants  = (L,1.0000001*L,1.0000002*L),
                units      = 'A',
                atoms      = atom,
                net_charge = v.q,
                net_spin   = v.q%2,
                kgrid      = (1,1,1),
                kshift     = (0,0,0)
                )
            s = system.structure
            s.slide(s.axes.sum(0)/2)
        #end if
        return system
    #end def make_system


    def checkv(self,v,required,tag):
        if len(set(required)-set(v.keys()))>0:
            self.error('cannot {0}, required variables are not present\n  variables required: {1}\n  variables present: {2}'.format(tag,required,v.keys()))
        #end if
    #end def checkv


    def get_ae_inputs(self,iq):
        inputs = obj()
        pinputs = obj()
        for name,value in self.inputs.iteritems():
            if name.startswith('ae_'):
                if name=='ae_occupations':
                    val = value[iq]
                else:
                    val = value
                #end if    
                inputs[name.replace('ae_','')] = val
                pinputs[name] = val
            #end if
        #end for
        for name,value in self.results.iteritems():
            if name.startswith('ae_'):
                inputs[name.replace('ae_','')] = value[iq]
                pinputs[name] = value[iq]
            #end if
        #end for 
        return inputs,pinputs
    #end def get_ae_inputs


    def get_pp_inputs(self,pp,iq):
        inputs = obj()
        pinputs = obj()
        for name,value in self.inputs.iteritems():
            if name.startswith('pp_'):
                inputs[name.replace('pp_','')] = value
                pinputs[name] = value
            #end if
        #end for
        for name,value in self.results.iteritems():
            if name.startswith('pp_'):
                inputs[name.replace('pp_','')] = value[pp][iq]
                pinputs[name] = value[pp][iq]
            #end if
        #end for 
        return inputs,pinputs
    #end def get_pp_inputs


    def make_all_sims(self,basepath):
        print '    make_all_sims',self.__class__.__name__,self.systype
        sinfo = self.system_info
        if self.systype=='ae':
            for iq in range(len(sinfo.q)):
                q = sinfo.q[iq]
                path = os.path.join(basepath,'q'+str(q))
                v,vp = self.get_ae_inputs(iq)
                self.check_inputs(vp)
                v.path = path
                v.q    = q
                sims = self.make_sims(v)
                self.save_sims(sims,q,vp,v)
            #end for
        elif self.systype=='pp':
            for functional,pplist in sinfo.func_pp.iteritems():
                for pp in pplist:
                    pseudos = sinfo.ppmap[pp]
                    for iq in range(len(sinfo.q)):
                        q = sinfo.q[iq]
                        path = os.path.join(basepath,functional,pp,'q'+str(q))
                        v,vp = self.get_pp_inputs(pp,iq)
                        self.check_inputs(vp)
                        v.path       = path
                        v.q          = q
                        v.pseudos    = pseudos
                        v.functional = functional
                        v.pp         = pp
                        sims = self.make_sims(v)
                        self.save_sims(sims,(functional,pp,q),vp,v)
                    #end for
                #end for
            #end for
        #end if
    #end def make_all_sims


    def get_result_sim(self,v,name):
        if self.systype=='ae':
            key = v.q
            name = 'ae_'+name
        elif self.systype=='pp':
            key = v.tuple('functional','pp','q')
            name = 'pp_'+name
        #end if
        sim = ValidationStage.get_result_sim(self,(name,key))
        return sim
    #end def get_result_sim


    def get_results(self,quantity=None):
        res = obj()
        sinfo = self.system_info
        if self.systype=='ae':
            for q in sinfo.q:
                key = q
                v = self.saved_inputs[key]
                sims = self.sims[key]
                res[key] = self.get_result(v,sims,quantity)
            #end for
        elif self.systype=='pp':
            for functional,pplist in sinfo.func_pp.iteritems():
                for pp in pplist:
                    for q in sinfo.q:
                        key = (functional,pp,q)
                        v = self.saved_inputs[key]
                        sims = self.sims[key]
                        res[key] = self.get_result(v,sims,quantity)
                    #end for
                #end for
            #end for
        #end if
        return res
    #end def get_results


    def make_sims(self,v):
        self.not_implemented()
    #end def make_sims


    def get_result(self,v,sims,name):
        self.not_implemented()
    #end def get_result
#end class AtomicValidationStage



class AtomicHFOccupationScan(AtomicValidationStage):
    systype = 'ae'
    stage_dependencies = set(['ae_hfjob'])

    data_type = 'scalar'
    title  = 'Hartree-Fock energies vs. orbital occupation'
    xlabel = 'Orbital occupations'
    ylabel = 'HF energy'
    yunit  = 'Ha'

    def make_sims(self,v):
        sims = obj()
        atom = self.make_system(v)
        for occ in v.occupations:
            up,down = occ
            path = os.path.join(v.path,'{0}__{1}'.format(up,down))
            path = path.replace(',','').replace('(','').replace(')','').replace(' ','')
            hf = generate_sqd(
                identifier = 'hf',
                path       = path,
                job        = v.hfjob,
                system     = atom,
                up         = up,
                down       = down,
                grid_type  = 'log',
                ri         = 1e-6 ,
                rf         = 400  ,
                npts       = 10001,
                max_iter   = 1000 ,
                etot_tol   = 1e-8 ,
                eig_tol    = 1e-12,
                mix_ratio  = 0.7  
                )
            sims[occ] = hf
        #end for
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        occs = []
        energies = []
        for occ in v.occupations:
            a = sims[occ].load_analyzer_image()
            energies.append(a.E_tot)
            occs.append(occ[0]+' '+occ[1])
        #end for
        return occs,energies
    #end def get_result
#end class AtomicHFOccupationScan




class AtomicHFCalc(AtomicValidationStage):
    systype = 'ae'
    stage_dependencies = set(['ae_hfjob','ae_occupation'])
    stage_result = 'ae_orbitals'
    final_result = 'Ehf_ae'

    stage_type = 'value'
    data_type  = 'scalar'
    title      = 'Hartree-Fock energy vs. Ion charge'
    ylabel     = 'HF energy'
    xlabel     = 'Ion charge'
    yunit      = 'Ha'

    def make_sims(self,v):
        sims = obj()
        atom = self.make_system(v)
        up,down = v.occupation
        hf = generate_sqd(
            identifier = 'hf',
            path       = v.path,
            job        = v.hfjob,
            system     = atom,
            up         = up,
            down       = down,
            grid_type  = 'log',
            ri         = 1e-6 ,
            rf         = 400  ,
            npts       = 10001,
            max_iter   = 1000 ,
            etot_tol   = 1e-8 ,
            eig_tol    = 1e-12,
            mix_ratio  = 0.7  
            )
        sims.orb = hf
        sims.result = hf
        return sims
    #end def make_sims


    def get_result(self,v,sims,name='E'):
        ha = sims.hf.load_analyzer_image()
        return ha.E
    #end def get_result
#end class AtomicHFCalc




class AtomicDFTBoxScan(AtomicValidationStage):
    systype = 'pp'
    stage_inputs       = set(['pp_Ls'])
    stage_dependencies = set(['pp_dftjob','pp_Ecut0'])

    data_type = 'scalar'
    title  = 'DFT energy vs. box size'
    ylabel = 'DFT energy'
    xlabel = 'box size'
    yunit  = 'Ry'
    xunit  = 'A'

    def make_sims(self,v):
        sims = obj()
        dftpp,qmcpp = sort_pseudos(v.pseudos)
        for L in v.Ls:
            atom = generate_physical_system(
                lattice    = 'orthorhombic',
                cell       = 'primitive',
                centering  = 'P',
                constants  = (L,1.0000001*L,1.0000002*L),
                units      = 'A',
                atoms      = self.system_info.atom,
                net_charge = v.q,
                net_spin   = v.q%2,
                kgrid      = (1,1,1),
                kshift     = (0,0,0)
                )
            s = atom.structure
            s.slide(s.axes.sum(0)/2)
            path = os.path.join(v.path,'L_'+str(L))
            scf = generate_pwscf(
                identifier   = 'scf',
                path         = path,
                job          = v.dftjob,
                input_type   = 'scf',
                input_dft    = v.functional,
                ecut         = v.Ecut0,
                conv_thr     = 1e-8,
                mixing_beta  = .7,
                nosym        = True,
                pseudos      = dftpp,
                system       = atom,
                use_folded   = False
                )
            sims[L] = scf
        #end for
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        Ls = []
        energies = []
        for L in v.Ls:
            pa = sims[L].load_analyzer_image()
            Ls.append(L)
            energies.append(pa.E)
        #end for
        return Ls,energies
    #end def get_result
#end class AtomicDFTBoxScan




class AtomicDFTEcutScan(AtomicValidationStage):
    systype = 'pp'
    stage_inputs       = set(['pp_Ecuts'])
    stage_dependencies = set(['pp_dftjob','pp_L'])

    data_type = 'scalar'
    title  = 'DFT energy vs. planewave energy cutoff'
    ylabel = 'DFT energy'
    xlabel = 'Ecut'
    yunit  = 'Ry'
    xunit  = 'Ry'

    def make_sims(self,v):
        sims = obj()
        dftpp,qmcpp = sort_pseudos(v.pseudos)
        atom = self.make_system(v)
        for ecut in v.Ecuts:
            path = os.path.join(v.path,'Ecut_'+str(ecut))
            scf = generate_pwscf(
                identifier   = 'scf',
                path         = path,
                job          = v.dftjob,
                input_type   = 'scf',
                input_dft    = v.functional,
                ecut         = ecut,
                conv_thr     = 1e-8,
                mixing_beta  = .7,
                nosym        = True,
                pseudos      = dftpp,
                system       = atom,
                use_folded   = False
                )
            p2c = generate_pw2casino(
                identifier   = 'p2c',
                path         = path,
                job          = Job(cores=1)
                )
            p2c.depends(scf,'orbitals')
            sims[ecut,'E'] = scf
            sims[ecut,'KE'] = p2c
        #end for
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        ecuts = []
        energies = []
        if not name in ['E','KE']:
            self.error('quantity {0} is not computed by {1}'.format(name,self.name))
        #end if
        for ecut in v.Ecuts:
            ecuts.append(ecut)
            pa = sims[ecut,name].load_analyzer_image()
            if name=='E':
                energies.append(pa.E)
            elif name=='KE':
                energies.append(pa.energies.Kinetic)
            #end if
        #end for
        return ecuts,energies
    #end def get_result
#end class AtomicDFTEcutScan




class AtomicDFTCalc(AtomicValidationStage):
    systype = 'pp'
    stage_dependencies = set(['pp_dftjob','pp_L','pp_Ecut'])
    stage_result = 'pp_orbitals'
    final_result = 'Edft_pp'

    stage_type = 'value'
    data_type  = 'scalar'
    title      = 'DFT energy vs Ion charge'
    ylabel     = 'DFT energy'
    xlabel     = 'Ion charge'
    yunit      = 'Ry'

    def make_sims(self,v):
        dftpp,qmcpp = sort_pseudos(v.pseudos)
        atom = self.make_system(v)
        scf = generate_pwscf(
            identifier   = 'scf',
            path         = v.path,
            job          = v.dftjob,
            input_type   = 'scf',
            input_dft    = v.functional,
            ecut         = v.Ecut,
            conv_thr     = 1e-8,
            mixing_beta  = .7,
            nosym        = True,
            pseudos      = dftpp,
            system       = atom,
            use_folded   = False
            )
        p2q = generate_pw2qmcpack(
            identifier   = 'p2q',
            path         = v.path,
            job          = Job(cores=1),
            write_psir   = False
            )
        p2q.depends(scf,'orbitals')
        sims = obj(
            scf = scf,
            p2q = p2q,
            result = p2q
            )
        return sims
    #end def make_sims


    def get_result(self,v,sims,name='E'):
        pa = sims.scf.load_analyzer_image()
        return pa.E
    #end def get_result
#end class AtomicDFTCalc




class AtomicOptJ1RcutScan(AtomicValidationStage):
    systype = 'pp'
    stage_inputs       = set(['pp_J1_rcuts','pp_opt_calcs'])
    stage_dependencies = set(['pp_optjob','pp_orbitals','pp_pade_b'])
    stage_result       = 'pp_J1_jastrow'

    title  = 'Optimal VMC Energy vs. J1 rcut'
    ylabel = 'Opt. Energy'
    xlabel = 'J1 rcut'
    yunit  = 'Ha'
    xunit  = 'B'

    def make_sims(self,v):
        sims = obj()
        res = obj()
        dftpp,qmcpp = sort_pseudos(v.pseudos)
        atom = self.make_system(v)
        orb  = self.get_result_sim(v,'orbitals')
        bu,bd = v.pade_b
        for rcut in v.J1_rcuts:
            path = os.path.join(v.path,'rcut_'+str(rcut))
            jastrows = [
                generate_jastrow('J1','bspline',8,rcut,system=atom),
                generate_jastrow('J2','pade',bu,bd,system=atom)
                ]
            opt = generate_qmcpack(
                identifier = 'opt',
                path       = path,
                job        = v.optjob,
                input_type = 'opt_jastrow',
                system     = atom,
                pseudos    = qmcpp,
                jastrows   = jastrows,
                opt_calcs  = v.opt_calcs
                )
            opt.depends(orb,'orbitals')
            sims[rcut] = opt
            res[rcut] = opt
        #end for
        sims.result = res
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name=None):
        rcuts    = []
        energies = []
        for rcut in v.J1_rcuts:
            qa = sims[rcut].load_analyzer_image()
            if 'vmc' in qa and len(qa.vmc)>0:
                series = array(qa.vmc.keys())
                qmc = qa.vmc[series.max()]
            else:
                series = array(qa.opt.keys())
                qmc = qa.opt[series.max()]
            #end if
            le = qmc.scalar.LocalEnergy
            rcuts.append(rcut)
            energies.append((le.mean,le.error))
        #end for
        return rcuts,energies
    #end def get_result
#end class AtomicOptJ1RcutScan




class AtomicOptCalc(AtomicValidationStage):

    title  = 'Opt. Energy vs. iteration #'
    ylabel = 'Opt. Energy'
    xlabel = 'iteration #'
    yunit  = 'Ha'

    def make_sims(self,v):
        atom = self.make_system(v)
        orb = self.get_result_sim(v,'orbitals')
        if self.systype=='ae':
            qmcpp = None
            bu,bd = v.pade_b
            jastrows = [
                generate_jastrow('J2','pade',bu,bd,system=atom),
                generate_jastrow('J3','polynomial',4,4,5.0,system=atom)
                ]
        elif self.systype=='pp':
            dftpp,qmcpp = sort_pseudos(v.pseudos)
            jastrows = []
        #end if
        opt = generate_qmcpack(
            identifier = 'opt',
            path       = v.path,
            job        = v.optjob,
            input_type = 'opt_jastrow',
            system     = atom,
            pseudos    = qmcpp,
            jastrows   = jastrows,
            opt_calcs  = v.opt_calcs
            )
        opt.depends(orb,'orbitals')
        if self.systype=='pp':
            J1_jastrows = self.get_result_sim(v,'J1_jastrow')
            if not v.J1_rcut in J1_jastrows:
                self.error('requested rcut {0} could not be found in J1 Jastrow rcuts\n  possible rcuts: {1}'.format(v.J1_rcut,J1_jastrows.keys()))
            #end if
            jsim = J1_jastrows[v.J1_rcut]
            opt.depends(jsim,'jastrow')
        #end if
        sims = obj(
            opt    = opt,
            result = opt
            )
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        qa = sims.opt.load_analyzer_image()
        series = []
        energies = []
        for s,opt in qa.opt.iteritems():
            series.append(s)
            le = opt.scalar.LocalEnergy
            energies.append((le.mean,le.error))
        #end for
        return series,energies
    #end def get_result
#end class AtomicOptCalc


class AtomicAEOptCalc(AtomicOptCalc):
    systype = 'ae'
    stage_inputs = set(['ae_opt_calcs'])
    stage_dependencies = set(['ae_optjob','ae_occupation','ae_orbitals','ae_pade_b'])
    stage_result = 'ae_jastrow'
#end class AtomicAEOptCalc


class AtomicPPOptCalc(AtomicOptCalc):
    systype = 'pp'
    stage_inputs = set(['pp_opt_calcs'])
    stage_dependencies = set(['pp_optjob','pp_L','pp_Ecut','pp_orbitals','pp_J1_rcut'])
    stage_result = 'pp_jastrow'
#end class AtomicPPOptCalc




class AtomicVMCCalc(AtomicValidationStage):

    stage_type = 'value'
    title  = 'VMC Energy vs. Ion charge'
    ylabel = 'VMC Energy'
    xlabel = 'Ion charge'
    yunit  = 'Ha'

    def make_sims(self,v):
        atom = self.make_system(v)
        orb = self.get_result_sim(v,'orbitals')
        jastrow = self.get_result_sim(v,'jastrow')
        if self.systype=='ae':
            qmcpp = None
        elif self.systype=='pp':
            dftpp,qmcpp = sort_pseudos(v.pseudos)
        #end if
        vmc = generate_qmcpack(
            identifier   = 'vmc',
            path         = v.path,
            job          = v.vmcjob,
            input_type   = 'basic',
            system       = atom,
            pseudos      = qmcpp,
            calculations = v.vmc_calcs
            )
        vmc.depends(orb,'orbitals')
        vmc.depends(jastrow,'jastrow')
        sims = obj(vmc=vmc)
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        qa = sims.vmc.load_analyzer_image()
        qmc = qa.qmc
        vmc = qmc[len(qmc)-1]
        le = vmc.scalar.LocalEnergy
        return (le.mean,le.error)
    #end def get_result
#end class AtomicVMCCalc


class AtomicAEVMCCalc(AtomicVMCCalc):
    systype = 'ae'
    stage_inputs = set(['ae_vmc_calcs'])
    stage_dependencies = set(['ae_vmcjob','ae_occupation','ae_orbitals','ae_jastrow'])
    final_result = 'Evmc_ae'
#end class AtomicAEVMCCalc


class AtomicPPVMCCalc(AtomicVMCCalc):
    systype = 'pp'
    stage_inputs = set(['pp_vmc_calcs'])
    stage_dependencies = set(['pp_vmcjob','pp_L','pp_Ecut','pp_orbitals','pp_jastrow'])
    final_result = 'Evmc_pp'
#end class AtomicPPVMCCalc




class AtomicDMCPopulationScan(AtomicValidationStage):

    title  = 'DMC Energy vs. Walker population'
    ylabel = 'DMC Energy'
    xlabel = 'Population'
    yunit  = 'Ha'

    def make_sims(self,v):
        sims = obj()
        atom = self.make_system(v)
        orb = self.get_result_sim(v,'orbitals')
        jastrow = self.get_result_sim(v,'jastrow')
        if self.systype=='ae':
            qmcpp = None
        elif self.systype=='pp':
            dftpp,qmcpp = sort_pseudos(v.pseudos)
        #end if
        for population in v.populations:
            calcs = deepcopy(v.dmc_calcs)
            found_vmc = False
            found_dmc = False
            for calc in calcs:
                if isinstance(calc,vmc):
                    if 'samplesperthread' in calc:
                        del calc.samplesperthread
                    #end if
                    calc.samples = population
                    found_vmc = True
                elif isinstance(calc,dmc):
                    found_dmc = True
                #end if
            #end for
            if not found_vmc or not found_dmc:
                self.error('vmc and dmc blocks must be present in dmc_calcs')
            #end if
            path = os.path.join(v.path,'pop_'+str(population))
            qmc = generate_qmcpack(
                identifier   = 'dmc',
                path         = path,
                job          = v.dmcjob,
                input_type   = 'basic',
                system       = atom,
                pseudos      = qmcpp,
                calculations = calcs
                )
            qmc.depends(orb,'orbitals')
            qmc.depends(jastrow,'jastrow')
            sims[population] = qmc
        #end if
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        pops = []
        energies = []
        for pop in v.populations:
            qa = sims[pop].load_analyzer_image()
            qmc = qa.qmc
            le = qmc[len(qmc)-1].dmc.LocalEnergy
            pops.append(pop)
            energies.append((le.mean,le.error))
        #end for
    #end def get_result
#end class AtomicDMCPopulationScan


class AtomicAEDMCPopulationScan(AtomicDMCPopulationScan):
    systype = 'ae'
    stage_inputs = set(['ae_dmc_calcs','ae_populations'])
    stage_dependencies = set(['ae_dmcjob','ae_occupation','ae_orbitals','ae_jastrow'])
#end class AtomicAEDMCPopulationScan


class AtomicPPDMCPopulationScan(AtomicDMCPopulationScan):
    systype = 'pp'
    stage_inputs = set(['pp_dmc_calcs','pp_populations'])
    stage_dependencies = set(['pp_dmcjob','pp_L','pp_orbitals','pp_jastrow'])
#end class AtomicPPDMCPopulationScan




class AtomicDMCTimestepScan(AtomicValidationStage):

    title  = 'DMC Energy vs. Timestep'
    ylabel = 'DMC Energy'
    xlabel = 'Timestep'
    yunit  = 'Ha'
    xunit  = '1/Ha'

    def make_sims(self,v):
        sims = obj()
        atom = self.make_system(v)
        orb = self.get_result_sim(v,'orbitals')
        jastrow = self.get_result_sim(v,'jastrow')
        if self.systype=='ae':
            qmcpp = None
        elif self.systype=='pp':
            dftpp,qmcpp = sort_pseudos(v.pseudos)
        #end if
        calcs = deepcopy(v.dmc_calcs)
        found_vmc = False
        found_dmc = False
        vmc_calc = None
        dmc_calc = None
        for calc in calcs:
            if isinstance(calc,vmc):
                if 'samplesperthread' in calc:
                    del calc.samplesperthread
                #end if
                calc.samples = v.population
                vmc_calc = calc
                found_vmc = True
            elif isinstance(calc,dmc):
                dmc_calc = calc
                found_dmc = True
            #end if
        #end for
        if not found_vmc or not found_dmc:
            self.error('vmc and dmc blocks must be present in dmc_calcs')
        #end if
        calcs = [vmc_calc]
        for timestep in v.timesteps:
            dcalc = dmc_calc.copy()
            tfac = dcalc.timestep/timestep
            dcalc.warmupsteps = int(round(tfac*dcalc.warmupsteps))
            dcalc.steps       = int(round(tfac*dcalc.steps))
            dcalc.timestep    = timestep
            calcs.append(dcalc)
        #end for
        qmc = generate_qmcpack(
            identifier   = 'dmc',
            path         = v.path,
            job          = v.dmcjob,
            input_type   = 'basic',
            system       = atom,
            pseudos      = qmcpp,
            calculations = calcs
            )
        qmc.depends(orb,'orbitals')
        qmc.depends(jastrow,'jastrow')
        sims.dmc = qmc
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        s=0
        timesteps = []
        energies  = []
        qa = sims.dmc.load_analyzer_image()
        dmc = qa.dmc
        for timestep in v.timesteps:
            s+=1
            le = dmc[s].dmc.LocalEnergy
            timesteps.append(timestep)
            energies.append((le.mean,le.error))
        #end for
        return timesteps,energies
    #end def get_result
#end class AtomicDMCTimestepScan


class AtomicAEDMCTimestepScan(AtomicDMCTimestepScan):
    systype = 'ae'
    stage_inputs = set(['ae_dmc_calcs'])
    stage_dependencies = set(['ae_dmcjob','ae_occupation','ae_orbitals','ae_jastrow','ae_population'])
#end class AtomicAEDMCTimestepScan


class AtomicPPDMCTimestepScan(AtomicDMCTimestepScan):
    systype = 'pp'
    stage_inputs = set(['pp_dmc_calcs'])
    stage_dependencies = set(['pp_dmcjob','pp_L','pp_orbitals','pp_jastrow','pp_population'])
#end class AtomicPPDMCTimestepScan




class AtomicDMCCalc(AtomicValidationStage):

    stage_type = 'value'
    title  = 'DMC Energy vs. Ion charge'
    ylabel = 'DMC Energy'
    xlabel = 'Ion charge'
    yunit  = 'Ha'

    def make_sims(self,v):
        atom = self.make_system(v)
        orb     = self.get_result_sim(v,'orbitals')
        jastrow = self.get_result_sim(v,'jastrow')
        if self.systype=='ae':
            qmcpp = None
        elif self.systype=='pp':
            dftpp,qmcpp = sort_pseudos(v.pseudos)
        #end if
        calcs = deepcopy(v.dmc_calcs)
        found_vmc = False
        found_dmc = False
        for calc in calcs:
            if isinstance(calc,vmc):
                if 'samplesperthread' in calc:
                    del calc.samplesperthread
                #end if
                calc.samples = v.population
                found_vmc = True
            elif isinstance(calc,dmc):
                calc.timestep = v.timestep
                found_dmc = True
            #end if
        #end for
        if not found_vmc or not found_dmc:
            self.error('vmc and dmc blocks must be present in dmc_calcs')
        #end if
        qmc = generate_qmcpack(
            identifier   = 'dmc',
            path         = v.path,
            job          = v.dmcjob,
            input_type   = 'basic',
            system       = atom,
            pseudos      = qmcpp,
            calculations = calcs
            )
        qmc.depends(orb,'orbitals')
        qmc.depends(jastrow,'jastrow')
        sims = obj(dmc=qmc)
        return sims
    #end def make_sims

        
    def get_result(self,v,sims,name='E'):
        qa = sims.dmc.load_analyzer_image()
        qmc = qa.qmc
        dmc = qmc[len(qmc)-1]
        le = dmc.dmc.LocalEnergy
        return (le.mean,le.error)
    #end def get_result
#end class AtomicDMCCalc


class AtomicAEDMCCalc(AtomicDMCCalc):
    systype = 'ae'
    stage_inputs = set(['ae_dmc_calcs'])
    stage_dependencies = set(['ae_dmcjob','ae_orbitals','ae_jastrow','ae_population','ae_timestep'])
    final_result = 'Edmc_ae'
#end class AtomicAEDMCCalc


class AtomicPPDMCCalc(AtomicDMCCalc):
    systype = 'pp'
    stage_inputs = set(['pp_dmc_calcs'])
    stage_dependencies = set(['pp_dmcjob','pp_orbitals','pp_jastrow','pp_population','pp_timestep'])
    final_result = 'Edmc_pp'
#end class AtomicPPDMCCalc





class ValidateAtomPP(ValidationProcess):

    allowed_stage_inputs = set(\
        ['ae_occupations', 'ae_opt_calcs', 'ae_vmc_calcs', 'ae_dmc_calcs', 
         'ae_populations', 'ae_timesteps', 
         'pp_Ls', 'pp_Ecuts', 'pp_opt_calcs', 'pp_J1_rcuts', 
         'pp_vmc_calcs', 'pp_populations', 'pp_timesteps', 'pp_dmc_calcs'])

    allowed_stage_results = set(\
        ['ae_hfjob', 'ae_optjob', 'ae_vmcjob', 'ae_dmcjob', 'ae_pade_b',
         'ae_occupation','ae_orbitals','ae_jastrow','ae_population','ae_timestep',
         'pp_dftjob', 'pp_optjob', 'pp_vmcjob', 'pp_dmcjob', 'pp_pade_b',
         'pp_Ecut0','pp_L','pp_Ecut','pp_orbitals','pp_J1_rcut','pp_jastrow',
         'pp_population','pp_timestep'])
        

    def initialize(self,**func_pp):
        if not 'q' in func_pp:
            self.error('variable q must be provided')
        #end if
        self.q = func_pp['q']
        del func_pp['q']
        ref = None
        if 'ref' in func_pp:
            ref = func_pp['ref']
            del func_pp['ref']
        #end if
        if 'name' in func_pp:
            self.name = func_pp['name']
            del func_pp['name']
        #end if
        self.check_funcs_pseudos(**func_pp)
        self.ref = ref

        self.add_stages(
            ae_occupation_scan = AtomicHFOccupationScan(),
            ae_hf_calc         = AtomicHFCalc(),
            ae_opt_calc        = AtomicAEOptCalc(),
            ae_vmc_calc        = AtomicAEVMCCalc(),
            ae_dmc_pop_scan    = AtomicAEDMCPopulationScan(),
            ae_dmc_tau_scan    = AtomicAEDMCTimestepScan(),
            ae_dmc_calc        = AtomicAEDMCCalc(),
            pp_box_scan        = AtomicDFTBoxScan(),
            pp_ecut_scan       = AtomicDFTEcutScan(),
            pp_dft_calc        = AtomicDFTCalc(),
            pp_J1_rcut_scan    = AtomicOptJ1RcutScan(),
            pp_opt_calc        = AtomicPPOptCalc(),
            pp_vmc_calc        = AtomicPPVMCCalc(),
            pp_dmc_pop_scan    = AtomicPPDMCPopulationScan(),
            pp_dmc_tau_scan    = AtomicPPDMCTimestepScan(),
            pp_dmc_calc        = AtomicPPDMCCalc()
            )

        self.set_stage_order(
            ['ae_occupation_scan','ae_hf_calc','ae_opt_calc','ae_vmc_calc',
             'ae_dmc_pop_scan','ae_dmc_tau_scan','ae_dmc_calc',
             'pp_box_scan','pp_ecut_scan','pp_dft_calc','pp_J1_rcut_scan',
             'pp_opt_calc','pp_vmc_calc','pp_dmc_pop_scan','pp_dmc_tau_scan',
             'pp_dmc_calc']
            )

        info = obj(
            q = self.q,
            ppmap = self.ppmap,
            func_pp = self.func_pp,
            functionals = self.functionals
            )
        for atom,stage in self.stages.iteritems():
            info.atom = self.name
            stage.system_info = info.copy()
        #end for

    #end def initialize
#end class ValidateAtomPP




class ValidateDimerPP(ValidationProcess):
    def initialize(self,**func_pp):
        self.check_funcs_pseudos(**func_pp)
    #end def __init__


    def dft_box(self):
        self.not_implemented()
    #end def dft_box


    def kinetic_ecut(self):
        self.not_implemented()
    #end def kinetic_ecut


    def kinetic_meshfactor(self):
        self.not_implemented()
    #end def kinetic_meshfactor


    def vmc_noJ(self):
        self.not_implemented()
    #end def vmc_noJ


    def opt_J1J2_rcut(self):
        self.not_implemented()
    #end def opt_J1J2_rcut


    def opt_J1J2J3(self):
        self.not_implemented()
    #end def opt_J1J2J3


    def dmc_population(self):
        self.not_implemented()
    #end def dmc_population


    def dmc_timestep(self):
        self.not_implemented()
    #end def dmc_timestep

#end class ValidateDimerPP




class ValidateAtomPPs(ValidationProcesses):
    def initialize(self):
        atoms = list(self.processes.keys())
        pseudos = obj()
        for atom in atoms:
            pseudos[atom] = self.processes[atom].pseudo_list()
        #end for
        pp_to_atom = obj()
        for atom,pplist in pseudos.iteritems():
            for pp in pplist:
                pp_to_atom[pp] = atom
            #end for
        #end for
        self.atoms = atoms
        self.pseudos = pseudos
        self.pp_to_atom = pp_to_atom
        nqset = set()
        nq = obj()
        charges = obj()
        for atom,process in self.processes.iteritems():
            nq[atom] = len(process.q)
            nqset.add(len(process.q))
            charges[atom] = list(process.q)
        #end for
        self.charges = charges
        self.nq = nq
        self.nq_same = len(nqset)==1
    #end def initialize


    def set_stage_inputs(self,**kwargs):
        if not 'ae_occupations' in kwargs:
            self.error('input variable ae_occupations must be provided')
        #end if
        ae_occupations = kwargs['ae_occupations']
        del kwargs['ae_occupations']
        for atom,process in self.processes.iteritems():
            if not atom in ae_occupations:
                self.error('atom {0} must be in ae_occupations'.format(atom))
            #end if
            inputs = obj(**kwargs)
            inputs.ae_occupations = ae_occupations[atom]
            process.set_stage_inputs(**inputs)
        #end for
    #end def set_stage_inputs


    def set_stage_results(self,**kwargs):
        ae_res = obj()
        pp_res = obj()
        for name,result in kwargs.iteritems():
            if name.startswith('ae_'):
                ae_res[name] = self.expand_ae_stage_result(result)
            elif name.startswith('pp_'):
                pp_res[name] = self.expand_pp_stage_result(result)
            else:
                self.error('invalid stage result variable encountered\n  variable encountered {0}\n  variable must be prefixed with ae_ or pp_'.format(name))
            #end if
        #end for

        #ae_comb = self.combine_ae_stage_results(ae_res)
        #pp_comb = self.combine_pp_stage_results(pp_res)

        for atom,process in self.processes.iteritems():
            res = obj()
            for name,rcoll in ae_res.iteritems():
                res[name] = deepcopy(rcoll[atom])
            #end for
            for name,rcoll in pp_res.iteritems():
                res[name] = deepcopy(rcoll[atom])
            #end for
            process.check_stage_results(res.keys())
            process.set_stage_results(**res)
        #end for
    #end def set_stage_results


    def expand_ae_stage_result(self,rin):
        r = obj()
        single_types = (str,int,float,tuple,type(None),Job)
        wrong_type = False
        if isinstance(rin,single_types):
            for atom in self.atoms:
                r[atom] = self.nq[atom]*[rin]
            #end for
        elif isinstance(rin,obj):
            atoms = set(rin.keys())
            if len(set(self.atoms)-atoms)>0:
                self.error('cannot expand stage result\n  information is required for all requested atoms\n  atoms requested: {0}\n  information provided:\n{1}'.format(self.atoms,rin))
            #end if
            for atom,ainfo in rin.iteritems():
                if atom in self.atoms:
                    ares = None
                    if isinstance(ainfo,list):
                        if not len(ainfo)==self.nq[atom]:
                            self.error('stage result list for atom {0} is not the same length as the charge list\n  result list: {1}\n  charge list: {2}'.format(atom,ainfo,self.processes[atom].q))
                        #end if
                        ares = list(ainfo)
                    elif isinstance(ainfo,single_types):
                        ares = self.nq[atom]*[ainfo]
                    else:
                        wrong_type = True
                    #end if
                    r[atom] = ares
                #end if
            #end for
        elif isinstance(rin,list):
            if not self.nq_same:
                self.error('charge lists are not the same length for each atom\n  stage results cannot be set by list provided\n  list provided: '+str(rin))
            #end if
            for atom in self.atoms:
                r[atom] = list(rin)
            #end for
        else:
            wrong_type = True
        #end if

        if wrong_type:
            self.error('cannot expand stage result\n  received type: {0}\n  allowed_types: {1}'.format(rin.__class__.__name__,single_types))
        #end if
        return r
    #end def expand_ae_stage_result


    def expand_pp_stage_result(self,rin):
        r = obj()
        single_types = (str,int,float,tuple,type(None),Job)
        wrong_type = False
        if isinstance(rin,single_types):
            for atom in self.atoms:
                ares = obj()
                for pp in self.pseudos[atom]:
                    ares[pp] = self.nq[atom]*[rin]
                #end for
                r[atom] = ares
            #end for
        elif isinstance(rin,obj):
            atoms = set(rin.keys())
            if len(set(self.atoms)-atoms)>0:
                self.error('cannot expand stage result\n  information is required for all requested atoms\n  atoms requested: {0}\n  information provided:\n{1}'.format(self.atoms,rin))
            #end if
            for atom,ainfo in rin.iteritems():
                if atom in self.atoms:
                    ares = obj()
                    if isinstance(ainfo,dict):
                        pps = set(ainfo.keys())
                        if len(set(self.pseudos[atom])-pps)>0:
                            self.error('cannot expand stage result\n  information is required for all requested pseudopotentials\n  pseudopotentials requested: {0}\n  information provided:\n{1}'.format(self.pseudos[atom],rin))
                        #end if
                        for pp,ppinfo in ainfo.iteritems():
                            if isinstance(ppinfo,list):
                                ares[pp] = list(ppinfo)
                            elif isinstance(ppinfo,single_types):
                                ares[pp] = self.nq[atom]*[ppinfo]
                            else:
                                wrong_type = True
                            #end if
                        #end for
                    elif isinstance(ainfo,list):
                        if not len(ainfo)==self.nq[atom]:
                            self.error('stage result list for atom {0} is not the same length as the charge list\n  result list: {1}\n  charge list: {2}'.format(atom,ainfo,self.processes[atom].q))
                        #end if
                        for pp in self.pseudos[atom]:
                            ares[pp] = list(ainfo)
                        #end for
                    elif isinstance(ainfo,single_types):
                        for pp in self.pseudos[atom]:
                            ares[pp] = self.nq[atom]*[ainfo]
                        #end for
                    else:
                        wrong_type = True
                    #end if
                    r[atom] = ares
                #end if
            #end for
        elif isinstance(rin,dict):
            pps = set(rin.keys())
            if len(set(self.pp_to_atom.keys())-pps)>0:
                self.error('cannot expand stage result\n  information is required for all requested pseudopotentials\n  pseudopotentials requested: {0}\n  information provided:\n{1}'.format(self.pp_to_atom.keys(),rin))
            #end if
            for pp,atom in self.pp_to_atom.iteritems():
                if pp in rin:
                    if not atom in r:
                        r[atom] = obj()
                    #end if
                    ppinfo = rin[pp]
                    ppres = obj()
                    if isinstance(ppinfo,list):
                        if not len(ppinfo)==self.nq[atom]:
                            self.error('stage result list for atom {0} is not the same length as the charge list\n  result list: {1}\n  charge list: {2}'.format(atom,ppinfo,self.processes[atom].q))
                        #end if
                        ppres = list(ppinfo)
                    elif isinstance(ppinfo,single_types):
                        ppres = self.nq[atom]*[ppinfo]
                    else:
                        wrong_type = True
                    #end if
                    r[atom][pp] = ppres
                #end if
            #end for
        elif isinstance(rin,list):
            if not self.nq_same:
                self.error('charge lists are not the same length for each atom\n  stage results cannot be set by list provided\n  list provided: '+str(rin))
            #end if
            for atom,pseudos in self.pseudos.iteritems():
                ares = obj()
                for pp in pseudos:
                    ares[pp] = list(rin)
                #end for
                r[atom] = ares
            #end for
        else:
            wrong_type = True
        #end if

        if wrong_type:
            self.error('cannot expand stage result\n  received type: {0}\n  allowed_types: {1}'.format(rin.__class__.__name__,single_types))
        #end if

        return r
    #end def expand_pp_stage_result


    def combine_ae_stage_results(self,res):
        # transpose the data
        comb = obj()
        for atom in self.atoms:
            acomb = obj()
            comb[atom] = acomb
            for iq in range(self.nq[atom]):
                q = self.charges[atom][iq]
                qcomb = obj()
                acomb[q] = qcomb
                for name,rcoll in res.iteritems():
                    qcomb[name.replace('ae_','')] = rcoll[atom][iq]
                #end for
            #end for
        #end for
        return comb
    #end def combine_ae_stage_results


    def combine_pp_stage_results(self,res):
        # transpose the data
        comb = obj()
        for atom in self.atoms:
            acomb = obj()
            comb[atom] = acomb
            for pp in self.pseudos[atom]:
                ppcomb = obj()
                acomb[pp] = ppcomb
                for iq in range(self.nq[atom]):
                    q = self.charges[atom][iq]
                    qcomb = obj()
                    ppcomb[q] = qcomb
                    for name,rcoll in res.iteritems():
                        qcomb[name.replace('pp_','')] = rcoll[atom][pp][iq]
                    #end for
                #end for
            #end for
        #end for
        return comb
    #end def combine_pp_stage_results
        
#end class ValidateAtomPPs
ValidateAtomPPs.set_process_type(ValidateAtomPP)




class ValidateDimerPPs(ValidationProcesses):
    def sim_list(self):
        return []
    #end def sim_list
#end class ValidateDimerPPs
ValidateDimerPPs.set_process_type(ValidateDimerPP)




class ValidatePPs(ValidatePPBase):
    def __init__(self,settings=None,atoms=None,dimers=None):
        if not isinstance(settings,Pobj):
            self.error('input variable settings must be the settings object from the Project Suite')
        #end if
        ValidatePPBase.settings = settings
        if atoms!=None:
            if not isinstance(atoms,(obj,dict)):
                self.error('input variable atoms must be of type dict or obj\n  you provided: '+atoms.__class__.__name__)
            #end if
            self.atoms = ValidateAtomPPs(**atoms)
        #end if
        if dimers!=None:
            if not isinstance(dimers,(obj,dict)):
                self.error('input variable dimers must be of type dict or obj\n  you provided: '+dimers.__class__.__name__)
            #end if
            self.dimers = ValidateDimerPPs(**dimers)
        #end if
        tests = obj()
        for name,test in self.iteritems():
            if test!=None:
                test.set_name(name)
                tests[name] = test
            #end if
        #end for
        self.tests = tests
    #end def __init__

            
    def sim_list(self):
        sims = []
        for test in self.tests:
            sims.extend(test.sim_list())
        #end for
        return sims 
    #end def sim_list


    def make_sims(self):
        print 'make_sims',self.__class__.__name__
        for name,test in self.tests.iteritems():
            test.make_sims(name)
        #end for
        return self.sim_list()
    #end def make_sims


    def status(self,*args):
        pad='  '
        n=0
        print 
        print n*pad+'ValidatePPs status'
        for test in self.tests:
            test.status(pad,n+1,*args)
        #end for
        print n*pad+'end ValidatePPs status'
        print
        if not 'continue' in args:
            exit()
        #end if
    #end def status
#end class ValidatePPs




def validate_qmc_pp(**kwargs):
    return ValidateQMCPPs(**kwargs)
#end def validate_qmc_pp




settings(
    pseudo_dir    = './pseudopotentials',
    status_only   = 0,
    generate_only = 0,
    sleep         = .3,
    machine       = 'node32'
    )



v = ValidatePPs(
    settings = settings,
    atoms = obj(
        #C = obj(
        #    q   = [0,1,2],
        #    lda = ['C.BFD'],
        #    pbe = ['C.BFD']
        #    ),
        O = obj(
            q   = [0,1],
            lda = ['O.6_lda']
            ),
        ),
    #dimers = obj(
    #    CO = obj(
    #        lda = ['C.BFD','O.6_lda'],
    #        pbe = ['C.BFD','O.6_lda']
    #        )
    #    )
    )

v.atoms.set_stage_inputs(
    ae_occupations = obj(     #varies w/ a,q
        C  = [[('He2s2p(-1,0)','He2s'),  # 0
               ('He2s2p(-1,1)','He2s'),
               ('He2s2p( 0,1)','He2s'),
               ('He2s2p(-1)'  ,'He2s2p(-1)'),
               ('He2s2p( 0)'  ,'He2s2p( 0)'),
               ('He2s2p( 1)'  ,'He2s2p( 1)'),
               ('He2s2p(-1)'  ,'He2s2p( 0)'),
               ('He2s2p(-1)'  ,'He2s2p( 1)')
               ],
              [('He2s2p(-1)'  ,'He2s'),  # 1
               ('He2s2p( 0)'  ,'He2s'),
               ('He2s2p( 1)'  ,'He2s'),
               ('He2s2p(-1,0)','He'),
               ('He2s2p(-1)'  ,'He2p(-1)')
               ],
              [('He2s'      ,'He2s'),    # 2
               ('He2s2p(-1)','He'),
               ('He2s'      ,'He2p(-1)')
               ]
              ],
        O  = [#[('He2s2p','He2s2p'),      # -2
              # ('He2s2p','He2s2p(-1,0)3s'),
              # ],
              #[('He2s2p','He2s2p(-1,0)'),# -1
              # ('He2s2p','He2s2p(-1,1)'),
              # ('He2s2p','He2s2p( 0,1)'),
              # ('He2s2p','He2p')
              # ],
              [('He2s2p','He2s2p(-1)'),  # 0
               ('He2s2p','He2s2p( 0)'),
               ('He2s2p','He2s2p( 1)'),
               ('He2s2p(-1,0)','He2s2p(-1,1)'),
               ('He2s2p(-1,1)','He2s2p(-1,0)'),
               ('He2s2p(-1,0)','He2s2p( 0,1)'),
               ('He2s2p( 0,1)','He2s2p(-1,0)'),
               ('He2s2p(-1,1)','He2s2p( 0,1)'),
               ('He2s2p( 0,1)','He2s2p(-1,1)')
               ],
              [('He2s2p','He2s'),        # 1
               ('He2s2p(-1,1)','He2s2p(0)'   ),
               ('He2s2p(-1,0)','He2s2p(1)'   ),
               ('He2s2p(-1)'  ,'He2s2p(0,1)' ),
               ('He2s2p(-1,0)','He2s2p(-1)'  ),
               ('He2s2p(-1,1)','He2s2p(-1)'  ),
               ('He2s2p(-1)'  ,'He2s2p(-1,0)'),
               ('He2s2p(-1)'  ,'He2s2p(-1,1)'),
               ]
              #[('He2s2p(-1,0)','He2s'),  # 2
              # ('He2s2p(-1,1)','He2s'),
              # ('He2s2p( 0,1)','He2s'),
              # ('He2s2p(-1)'  ,'He2s2p(-1)'),
              # ('He2s2p( 0)'  ,'He2s2p( 0)'),
              # ('He2s2p( 1)'  ,'He2s2p( 1)'),
              # ('He2s2p(-1)'  ,'He2s2p( 0)'),
              # ('He2s2p(-1)'  ,'He2s2p( 1)')
              # ]
              ]
        ),
    pp_Ls          = [10,15,20],             #same for all a,q,pp
    pp_Ecuts       = [100,150,200,250,300],  #same for all a,q,pp
    pp_J1_rcuts    = [3,4,5,6,7],            #same for all a,q,pp
    ae_populations = [500,1000,2000],        #same for all a,q
    pp_populations = [500,1000,2000],        #same for all a,q,pp
    ae_timesteps   = [.008,.004,.002,.001],  #same for all a,q
    pp_timesteps   = [.08,.04,.02,.01,.005], #same for all a,q,pp
    ae_opt_calcs   = [
        loop(max=4,
             qmc = linear(
                energy               = 0.0,
                unreweightedvariance = 1.0,
                reweightedvariance   = 0.0,
                walkers              = 1,
                warmupsteps          = 200,
                blocks               = 500,
                timestep             = 0.1,
                usedrift             = True,
                samples              = 16000,
                stepsbetweensamples  = 100,
                minwalkers           = 0.0,
                bigchange            = 15.0,
                alloweddifference    = 1e-4
                )
             ),
        loop(max=4,
             qmc = linear(
                energy               = 0.5,
                unreweightedvariance = 0.0,
                reweightedvariance   = 0.5,
                walkers              = 1,
                warmupsteps          = 200,
                blocks               = 500,
                timestep             = 0.1,
                usedrift             = True,
                samples              = 32000,
                stepsbetweensamples  = 100,
                minwalkers           = 0.0,
                bigchange            = 15.0,
                alloweddifference    = 1e-4
                )
             )
        ],
    ae_vmc_calcs   = [
        vmc(
            walkers     = 1,   
            warmupsteps = 50,
            blocks      = 1000,
            steps       = 10,
            substeps    = 10,
            timestep    = 0.1,
            usedrift    = True
            )
        ],
    ae_dmc_calcs   = [
        vmc(
            walkers     = 1,   
            warmupsteps = 20,
            blocks      = 100,
            steps       = 10,
            substeps    = 10,
            timestep    = 0.1,
            usedrift    = True,
            samples     = 1000
            ),
        dmc(
            warmupsteps = 24,
            blocks      = 1000,
            steps       = 12,
            timestep    = 0.004
            ) 
        ],
    pp_opt_calcs   = [
        loop(max=4,
             qmc = linear(
                energy               = 0.0,
                unreweightedvariance = 1.0,
                reweightedvariance   = 0.0,
                walkers              = 1,
                warmupsteps          =  50,
                blocks               = 500,
                timestep             = 0.4,
                usedrift             = True,
                samples              = 16000,
                stepsbetweensamples  = 10,
                minmethod            = 'quartic',
                minwalkers           = 0.5,
                bigchange            = 2.0,
                alloweddifference    = 1e-4,
                maxweight            = 1e9,
                beta                 = 0.025,
                exp0                 = -16,
                stepsize             = 0.2,
                stabilizerscale      = 1.0,
                nstabilizers         = 3,
                nonlocalpp           = True,
                usebuffer            = True
                )
             ),
        loop(max=4,
             qmc = linear(
                energy               = 0.5,
                unreweightedvariance = 0.0,
                reweightedvariance   = 0.5,
                walkers              = 1,
                warmupsteps          =  50,
                blocks               = 500,
                timestep             = 0.4,
                usedrift             = True,
                samples              = 32000,
                stepsbetweensamples  = 10,
                minmethod            = 'quartic',
                minwalkers           = 0.5,
                bigchange            = 2.0,
                alloweddifference    = 1e-4,
                maxweight            = 1e9,
                beta                 = 0.025,
                exp0                 = -16,
                stepsize             = 0.2,
                stabilizerscale      = 1.0,
                nstabilizers         = 3,
                nonlocalpp           = True,
                usebuffer            = True
                )
             )
        ],
    pp_vmc_calcs   = [
        vmc(
            walkers     = 1,   
            warmupsteps = 50,
            blocks      = 1000,
            steps       = 20,
            substeps    =  3,
            timestep    = 0.4,
            usedrift    = True
            )
        ],
    pp_dmc_calcs   = [
        vmc(
            walkers     = 1,   
            warmupsteps = 50,
            blocks      = 1000,
            steps       = 20,
            substeps    =  3,
            timestep    = 0.4,
            usedrift    = True,
            samples     = 1000,
            ),
        dmc(
            warmupsteps = 12,
            blocks      = 1000,
            steps       = 4,
            timestep    = 0.04,
            nonlocalmoves = True
            ) 
        ] 
    )


v.atoms.set_stage_results(
    ae_hfjob      = Job(cores=1),
    ae_optjob     = Job(cores=16),
    ae_vmcjob     = Job(cores=16),
    ae_dmcjob     = Job(cores=16),
    pp_dftjob     = Job(cores=16), 
    pp_optjob     = Job(cores=16),
    pp_vmcjob     = Job(cores=16),
    pp_dmcjob     = Job(cores=16),
    pp_Ecut0      = 150,   
    #ae_occupation = obj(   
    #    C         = [('He2s2p(-1,0)','He2s'        ),
    #                 ('He2s2p(-1)'  ,'He2s'        ),
    #                 ('He2s'        ,'He2s'        )],
    #    O         = [('He2s2p'      ,'He2s2p(-1,0)'),
    #                 ('He2s2p'      ,'He2s2p(-1)'  ),
    #                 ('He2s2p'      ,'He2s'        ),
    #                 ('He2s2p(-1,0)','He2s'        )]
    #    ),
    #pp_L          = 20,    
    #pp_Ecut = obj(
    #    C         = 150,
    #    O         = 200
    #    ),   
    #ae_orbitals   = 'finished',
    #pp_orbitals   = 'finished',
    #ae_pade_b     = (4.,4.),
    #pp_pade_b     = (4.,4.),
    #pp_J1_rcut = obj( 
    #    C         = {'C.BFD' : [5, 4, 3]},
    #    O         = 4
    #    ),
    #ae_jastrow    = 'finished',
    #pp_jastrow    = 'finished',
    #ae_population = obj(
    #    C         = 1000,  
    #    O         = 2000
    #    ),
    #pp_population = obj(
    #    C         = [2000,3000,4000], 
    #    O         = [1500,2500,3500,4500]
    #    ),
    #ae_timestep   = 1e-3, 
    #pp_timestep = {     
    #    'C.BFD'   : .01,
    #    'O.6_lda' : .02
    #    }
    )







# make the simulations
sims = v.make_sims()



#v.status()


pm = ProjectManager()

pm.add_simulations(sims)

pm.run_project()


#v.atoms.show_results(
#    stages = 'pp_box_scan',
#    mode   = 'print',
#    units  = 'eV',
#    xunits = 'B'
#    )


v.atoms.show_results(
    stages = 'ae_occupation_scan',
    mode   = 'print'
    )










#v = ValidatePPs(
#    settings = settings,
#    atoms = obj(
#        Cu = obj(
#            ref = (0,'lda','Cu.pz'),
#            q   = [0,1,2],
#            lda = ['Cu.17_lda','Cu.17_lda_f','Cu.19_lda_opt'],
#            pbe = ['Cu.17_lda']
#            ),
#        Zn = obj(
#            q   = [0,1,2],
#            lda = ['Zn.lda.pp5','Zn.lda.pp6'],
#            pbe = ['Zn.lda.pp6']
#            ),
#        #Co = obj(
#        #    q = [0,1,2],
#        #    pbe = ['Co.pbe'],
#        #    hse = ['Co.pbe','Co.pbe2']
#        #    )
#        ),
#    #dimers = obj(
#    #    CuO = obj(
#    #        lda = ['Cu.pz','O.pz'],
#    #        pbe = ['Cu.pbe','O.pbe']
#    #        ),
#    #    ZnO = obj(
#    #        lda = ['Zn.pz','Zn.pz']
#    #        ),
#    #    CoO = obj(
#    #        lda = ['Co.pz','O.pz'],
#    #        pbe = ['Co.pbe','O.pbe']
#    #        )
#    #    )
#    )
#
#v.atoms.set_stage_inputs(
#    hfjob  = Job(cores=1),
#    dftjob = Job(cores=16), 
#    optjob = Job(cores=16),
#    vmcjob = Job(cores=16),
#    dmcjob = Job(cores=16),
#    ae_occupations = obj(     #varies w/ a,q
#        Cu = [('abc1','def')],
#        Zn = [('abc2','def')],
#        Co = [('abc3','def')]
#        ),
#    ae_opt_calcs       = [],    #same for all a,q
#    ae_vmc_calcs       = [],    #same for all a,q
#    ae_populations     = [],    #same for all a,q
#    ae_timesteps       = [],    #same for all a,q
#    ae_dmc_calcs       = [],    #same for all a,q
#    pp_Ls              = [],    #same for all a,q,pp
#    pp_Ecuts           = [],    #same for all a,q,pp
#    pp_opt_calcs       = [],    #same for all a,q,pp
#    pp_J1_rcuts        = [],    #same for all a,q,pp
#    pp_vmc_calcs       = [],    #same for all a,q,pp
#    pp_dmc_populations = [],    #same for all a,q,pp
#    pp_dmc_timesteps   = [],    #same for all a,q,pp
#    pp_dmc_calcs       = []     #same for all a,q,pp
#    )
#
##v.atoms.set_stage_results(
##    ae_occupation = obj(      #varies w/ a,q
##        Cu = [],
##        Zn = [],
##        Co = []
##        ),
##    ae_population = 1000,     #can vary or be same
##    ae_timestep   = .0001,    #can vary or be same
##    pp_L          = 20,       #can vary or be same
##    pp_Ecut       = 300,      #will vary w/ a,pp
##    pp_J1_rcut    = obj(      #will vary w/ a,q,pp
##        Cu = {'Cu.pz' : [3.4, 4.5, 3.6],
##              'Cu.pbe': [3.5, 3.4, 4.2]},
##        Co = {}
##        ),
##    pp_population = 2000,     #can vary or be same
##    pp_timestep   = .01       #can vary or be same
##    )
#
#v.atoms.set_stage_results(
#    ae_occupation = obj(      #varies w/ a,q
#        Cu = [('up','down'),('up','down'),('up','down')],
#        Zn = [('up','down'),('up','down'),('up','down')],
#        Co = []
#        ),
#    ae_orbitals = 'finished',
#    ae_jastrow  = 'finished',
#    ae_population = obj(
#        Cu = 1000,     #can vary or be same
#        Zn = 2000
#        ),
#    ae_timestep   = [1e-4,2e-4,3e-4],    #can vary or be same
#    pp_Ecut0      = 200,   #same for all a,q,pp
#    pp_L          = 20,       #can vary or be same
#    pp_Ecut       = obj(
#        Cu = 150,
#        Zn = 200
#        ),      #will vary w/ a,pp
#    pp_orbitals = 'finished',
#    pp_jastrow  = 'finished',
#    pp_J1_rcut    = obj(      #will vary w/ a,q,pp
#        Cu = {'Cu.17_lda'    : [3.4, 4.5, 3.6],
#              'Cu.17_lda_f'  : [3.5, 3.4, 4.2],
#              'Cu.19_lda_opt': 5.0  },
#        Zn = 2.7,
#        Co = {}
#        ),
#    pp_population = obj(
#        Cu = [2000,3000,4000],     #can vary or be same
#        Zn = [2500,3500,4500]
#        ),
#    pp_timestep   = {       #can vary or be same
#        'Cu.17_lda'    :.01,
#        'Cu.17_lda_f'  :.02,
#        'Cu.19_lda_opt':.001,
#        'Zn.lda.pp5':.005,
#        'Zn.lda.pp6':.006
#        }
#    )
#




#  
# todo list 
#   qmcpack.py
#     don't add pp files to transfer files if pseudos is None
#   sqd_analyzer.py
#     needs to read log file for kinetic and total energies
#   ensure all pps are requested properly
#   update stage_dependencies
#   update stage initialization PP/AE in AtomicValidationProcess
#   update allowed stage results 
#   try out with generate_only
#   add dimer code
#   try w/ generate_only
#   walk through C,O and CO
#     make sure qmcpack was made w/ buildlevel=2
#   add tools directory
#     pseudopotential testing
#  
