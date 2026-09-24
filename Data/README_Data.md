### Data/ directory

Since the files used in the analysis take up a lot of memory, the queries submitted to download the same files using the Gaia Archive (ADQL, [https://gea.esac.esa.int/archive/](https://gea.esac.esa.int/archive/)) are provided below:


# BootesI_3.csv

SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pm, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error

FROM gaiadr3.gaia_source

WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
		CIRCLE('ICRS',210.0200,+14.5135210,3.0))


# BootesIII_10.csv

SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pm, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag

FROM gaiadr3.gaia_source

WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
		CIRCLE('ICRS',209.3,26.8,10.0))
		

# ComaBerenices_3.csv

SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pm, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error

FROM gaiadr3.gaia_source

WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
		CIRCLE('ICRS',186.7454,23.9069,3.0))
		

# CarinaII_3.csv

SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pm, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error

FROM gaiadr3.gaia_source

WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
		CIRCLE('ICRS',114.1066,-57.9991,3.0))
		
		
# SagittariusII_3.csv
		
SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pm, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error

FROM gaiadr3.gaia_source

WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
		CIRCLE('ICRS',298.1663,-22.065,3.0))
		
		
# UrsaMajorI_3.csv

SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pm, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error

FROM gaiadr3.gaia_source

WHERE 1=CONTAINS(POINT('ICRS',ra,dec),
		CIRCLE('ICRS',158.7706,51.9479,3.0))

		
# vari_rrlyrae.csv
	
SELECT *

FROM gaiadr3.vari_rrlyrae


# vari_classifier_result.csv

SELECT source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pm, pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error

FROM gaiadr3.gaia_source

JOIN gaiadr3.vari_classifier_result

using (source_id)

where best_class_name= 'RR'


# flux.csv

SELECT source_id, phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error

FROM gaiadr3.gaia_source

JOIN gaiadr3.vari_rrlyrae

using (source_id)
