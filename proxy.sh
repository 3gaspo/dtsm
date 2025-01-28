#. ./"proxy.sh"

#! /bin/bash

NNI=$1
if [ "$NNI" = "" ]; then
  echo "NNI ??"
  read NNI
fi
shift
PASS=$1
if [ "$PASS" = "" ]; then
  echo -n "Sesame pour $NNI ?? "
  read -s PASS
fi
export http_proxy=""
export https_proxy=""

EDFPROXY=vip-users.proxy.edf.fr:3128

unset HTTP_PROXY HTTPS_PROXY

curl --help > /dev/null 2>&1
ISCURL=$?
wget --help > /dev/null 2>&1
ISWGET=$?
NOEXPORT=0

if [ $ISCURL -eq 0 ]; then
  LOCATION=`curl -k -s -I --user "$NNI:$PASS" -x $EDFPROXY http://nonexistentdomain.domain | awk '
                             BEGIN { res="KO" }
                             NR==1 && $2 == "502" { res="OK" }
                             /^Location/          { res=$2 }
                             END { print res } ' | tr -d "\r"`
  if [ "$LOCATION" == "OK" ];then
    echo "Already authentified."
  elif [ "$LOCATION" == "KO" ];then
    echo $LOCATION
    echo "Probleme d'authentification sur le proxy $EDFPROXY"
    NOEXPORT=1
  else
    curl -k -I --user "$NNI:$PASS" --noproxy '*' "$LOCATION"
  fi
elif [ $ISWGET -eq 0 ]; then
  WGETOPT="-e use_proxy=yes -e http_proxy=$EDFPROXY --user=$NNI --password=$PASS -q --no-check-certificate -S -O -"
  LOCATION=`wget $WGETOPT http://nonexistentdomain.domain 2>&1 | awk '
                             BEGIN { res="KO" }
                             $1 ~ /HTTP\/1\.1/ && $2 == "502" { res="OK" }
                             END { print res } '`
   if [ "$LOCATION" == "OK" ];then
         echo "Authentified."
   else
      echo "Probleme d'authentification sur le proxy $EDFPROXY"
      NOEXPORT=1
   fi
else
  echo "NI curl, NI wget n'existe dans le PATH de cette machine."
  NOEXPORT=1
fi

if [ $NOEXPORT -eq 0 ]; then
  echo ""
    echo "Si vous n'avez pas sourcé (lancement avec un . et un espace devant) ce script vous devez copier/coller ..."
    echo export http_proxy=http://$EDFPROXY
    export http_proxy=http://$EDFPROXY
    echo export https_proxy=http://$EDFPROXY
    export https_proxy=http://$EDFPROXY
    echo export ftp_proxy=http://$EDFPROXY
    export ftp_proxy=http://$EDFPROXY
    echo export no_proxy=.edf.fr
    export no_proxy=.edf.fr

    echo "## pour R ##"
    echo "############"
    echo "Sys.setenv(http_proxy = 'http://$EDFPROXY')"
    echo "Sys.setenv(https_proxy = 'http://$EDFPROXY')"
    echo "Sys.setenv(ftp_proxy = 'http://$EDFPROXY')"
    echo "Sys.setenv(no_proxy = '.edf.fr')"
    echo "## pour Python ##"
    echo "#################"
    echo "os.environ['http_proxy']='http://$EDFPROXY'"
    echo "os.environ['http_proxy']='http://$EDFPROXY'"
    echo "os.environ['ftp_proxy']='http://$EDFPROXY'"
    echo "os.environ['no_proxy']='.edf.fr'"
fi


