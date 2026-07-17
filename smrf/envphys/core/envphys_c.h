/* From topotherm.c */
void topotherm(
    int ngrid, double *ta, double *tw, double *z, double *skvfac, int nthreads, double *thermal
);
double saturation_vapor_pressure(double *ta);
double satw(double tk);
double sati(double tk);
double brutsaert(double ta, double lmba, double ea, double z, double pa);

/* from dewpt.c */
void dew_point_t_for_grid(int ngrid, double *ea, double *dpt, int nthreads, double tolerance);
double dew_point_temperature(double vp_current, double tolerance);
double
zero_break(double lower_temperature, double upper_temperature, double vp_current, double tolerance);

/* from iwbt.c */
void iwbt(int ngrid, double *ta, double *td, double *z, int nthreads, double tol, double *tw);
double wetbulb(double ta, double dpt, double press, double tol);
