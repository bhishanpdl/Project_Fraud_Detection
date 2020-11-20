# Create new branch
```bash
# create new branch and go to it
git checkout -b new-branch

# the new branch has all files from current branch
echo "this is new branch" > new.txt
git add --all
git commit -m "created new branch"
git push origin new-branch  #***** NOT origin master *****

Go to github website and check it.



To check branches
git branch

To go to another branch
git checkout my-branch  # eg. master branch1 branch2

```
