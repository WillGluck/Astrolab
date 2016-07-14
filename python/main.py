import sys

sys.path.insert(0, 'core/')
sys.path.insert(0, 'data/')
sys.path.insert(0, 'general/')

from astrolab_main import AstrolabMain

main = AstrolabMain()
main.start()
