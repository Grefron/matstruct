drop table if exists isotropic;

create table isotropic
(
  id integer primary key not null,
  name text not null,
  modulus real,
  shearmodulus real,
  poissonratio real,
  density real not null,
  tensilestrength real,
  compressivestrength real,
  shearstrength real,
  thermalcoefficient real
);

insert into isotropic ('name', 'modulus', 'shearmodulus', 'poissonratio', 'density' ) values
  ('polyester', 3e9, 1e9, 0.35, 1200),
  ('epoxy', 4e9, 1e9, 0.35, 1100),
  ('pu35', 100e6, 50e6, 0.3, 35),
  ('eglass', 78e9, 20e9, 0.25, 1550),
  ('hscarbon', 230e9, 40e9, 0.2, 1500),
  ('hmcarbon', 440e9, 40e9, 0.2, 1500);
