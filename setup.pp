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

	
