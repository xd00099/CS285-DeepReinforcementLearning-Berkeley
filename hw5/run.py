import shlex, subprocess

command_stem = [
"python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0   --exp_name q5_iql_easy_supervised_lam{l}_tau{t}                                --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
"python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0   --exp_name q5_iql_easy_unsupervised_lam{l}_tau{t}   --unsupervised_exploration --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
"python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam{l}_tau{t}                              --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
"python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam{l}_tau{t} --unsupervised_exploration --use_rnd --num_exploration_steps=20000 --awac_lambda={l} --iql_expectile={t}",
]

awac_l = [10, 10, 2, 0.1] # easy-sup, easy-unsup, medium-sup, medium-unsup
iql_tau = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

commands = []
for i in range(4):
    command = command_stem[i]
    for tau in iql_tau:
        commands.append(command.format(l=awac_l[i], t=tau))

if __name__ == "__main__":
    for command in commands:
        print(command)
    # user_input = None
    # while user_input not in ['y', 'n']:
    #     user_input = input('Run experiment with above commands? (y/n): ')
    #     user_input = user_input.lower()[:1]
    # if user_input == 'n':
    #     exit(0)
    # for command in commands:
    #     args = shlex.split(command)
    #     subprocess.Popen(args)