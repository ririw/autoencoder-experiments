package {'build-essential': ensure => 'installed'}
package {'gfortran': ensure => 'installed'}
package {'libatlas3-base': ensure => 'installed'}
package {'libatlas-dev': ensure => 'installed'}
package {'libblas-dev': ensure => 'installed' }
package {'libfreetype6-dev': ensure => 'installed'}
package {'liblapack-dev': ensure => 'installed'}
package {'libpng12-dev': ensure => 'installed'}
package {'python-dev': ensure => 'installed'}
package {'python-numpy': ensure => 'installed' }
package {'python-pil': ensure => 'installed' }
package {'python-scipy': ensure => 'installed' }

$rootpath="/home/riri/Documents/autoencoders"
$user="riri"

vcsrepo { "$rootpath/pylearn2":
  ensure   => present,
  provider => git,
  source   => "git://github.com/lisa-lab/pylearn2.git",
  user     => $user,
}->
exec{"pip install":
  command => "$rootpath/env/bin/python $rootpath/pylearn2/setup.py develop",
  cwd     => "$rootpath/pylearn2",
  user    => $user
}

	
