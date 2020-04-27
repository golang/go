# Assumes that you've committed your work on your current branch
#
# OSX must be installed brew first
# Debian and Ubuntu repositories. Install using sudo apt-get install jq.
# Fedora repository. Install using sudo dnf install jq.
# openSUSE repository. Install using sudo zypper install jq.
# repository. Install using sudo pacman -Sy jq.
#

function sync_fork() {

orgin_git=""

DIRFILES=`ls $1`
if [[ $DIRFILES =~ "package.json" ]]
then
    # Verify that the command line Json parser is installed.
    if hash jq 2>/dev/null;
        then
           echo "jq installed!!"
        else
           if hash brew 2>/dev/null;
             then
                brew install jq
             else
                echo "Please refer to the comment installation for the jq tool in the corresponding OS."
           fi
    fi

    #Get the source address in the package. Json.
    orgin_git=$(cat package.json | jq '.repository.url')
    orgin_git=${orgin_git##*+}
    orgin_git=${orgin_git%%"\""}
  else
     echo -n  "please enter the source repository url ->  "
     read  orgin_git

 fi



#Syncing a Fork with the main repository
function sync_source() {
   current_branch=$(git rev-parse --abbrev-ref HEAD)

   git fetch upstream
   git checkout master
   git commit -m 'update'
   git merge upstream/master
   git push # origin
   git checkout $current_branch
}


#Check if the local repository exists upstream.
#If it exists, it is in direct sync.
#If not, add upstream to fork.
my_remote_repository=$(git remote -v)
echo $my_remote_repository
if [[ $my_remote_repository =~ "upstream" ]]
then
   sync_source
else
   git remote add upstream $orgin_git
   sync_source
fi
}

sync_fork