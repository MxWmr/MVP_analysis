from MVPConverter import Converter


mvpa = Converter('/home/maxw/Documents/ESSTECH25/MVP300_DATA/Stationary_Profiling/',subdirs=True)

# mvpa.help()

mvpa.load_mvp_data() 

# mvpa.load_ctd_data('/home/maxw/Documents/ESSTECH25/BATHYSONDE/DATA/TRAIT/CNV/')


mvpa.mvp_to_netcdf(compression=True, per_profile_files=False)   
# mvpa.ctd_to_netcdf(compression=True, per_profile_files=True)


