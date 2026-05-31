#!/usr/bin/env python3
"""TEM training with GPU support."""

import argparse
import os, glob, shutil
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import world
import utils
import parameters
import model


def _move_hyper_tensors(hyper, device):
    """Move all tensors nested in hyper dict/lists to device."""
    for key, val in hyper.items():
        if isinstance(val, torch.Tensor):
            hyper[key] = val.to(device)
        elif isinstance(val, (list, tuple)):
            hyper[key] = [_move_tensor_or_keep(v, device) for v in val]
        elif isinstance(val, dict):
            _move_hyper_tensors(val, device)


def _move_tensor_or_keep(val, device):
    if isinstance(val, torch.Tensor):
        return val.to(device)
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-it', type=int, default=20000)
    parser.add_argument('--env', type=str, default='./envs/5x5.json')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda')

    run_path, train_path, model_path, save_path, script_path, envs_path = utils.make_directories()
    for f in glob.iglob(os.path.join('.', '*.py')):
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(script_path, f))

    params = parameters.parameters()
    params['train_it'] = args.train_it

    tem = model.Model(params)
    tem = tem.to(device)
    _move_hyper_tensors(tem.hyper, device)

    writer = SummaryWriter(train_path)
    logger = utils.make_logger(run_path)
    adam = torch.optim.Adam(tem.parameters(), lr=params['lr_max'])

    envs = [args.env]
    environments = []
    visited = []
    walks = []
    prev_iter = None

    for _ in range(params['batch_size']):
        env = world.World(envs[0], randomise_observations=True)
        environments.append(env)
        visited.append([False] * env.n_locations)
        walks.append(env.generate_walks(
            params['n_rollout'] * np.random.randint(params['walk_it_min'], params['walk_it_max']), 1
        )[0])

    logger.info(f'Training {params["train_it"]} iters, batch={params["batch_size"]}, rollout={params["n_rollout"]}, device={device}')
    print(f'Training {params["train_it"]} iters, batch={params["batch_size"]}, rollout={params["n_rollout"]}, device={device}')

    for i in range(0, params['train_it']):
        t0 = time.time()
        eta_new, lambda_new, p2g_scale_offset, lr, walk_length_center, loss_weights = parameters.parameter_iteration(i, params)
        tem.hyper['eta'] = eta_new
        tem.hyper['lambda'] = lambda_new
        tem.hyper['p2g_scale_offset'] = p2g_scale_offset
        for pg in adam.param_groups:
            pg['lr'] = lr

        chunk = []
        for env_i, walk in enumerate(walks):
            if len(walk) < params['n_rollout']:
                environments[env_i] = world.World(envs[0], randomise_observations=True)
                visited[env_i] = [False] * environments[env_i].n_locations
                walk = environments[env_i].generate_walks(
                    params['n_rollout'] * np.random.randint(
                        int(walk_length_center - params['walk_it_window'] * 0.5),
                        int(walk_length_center + params['walk_it_window'] * 0.5)
                    ), 1
                )[0]
                walks[env_i] = walk
                if prev_iter is not None:
                    prev_iter[0].a[env_i] = None
            for _ in range(params['n_rollout']):
                if len(chunk) < params['n_rollout']:
                    chunk.append([[comp] for comp in walk.pop(0)])
                else:
                    for ci, comp in enumerate(walk.pop(0)):
                        chunk[_][ci].append(comp)
        for step in chunk:
            step[1] = torch.stack(step[1], dim=0).to(device)

        forward = tem(chunk, prev_iter)

        loss = torch.tensor(0.0, device=device)
        for step in forward:
            step_loss_list = []
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]['id']]:
                    step_loss_list.append(loss_weights.to(device) * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]['id']] = True
            if step_loss_list:
                loss = loss + torch.sum(torch.mean(torch.stack(step_loss_list, dim=0), dim=0))

        adam.zero_grad()
        if loss.item() != 0:
            loss.backward(retain_graph=True)
            adam.step()
        prev_iter = [forward[-1].detach()]

        acc_vals = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0) * 100
        elapsed = time.time() - t0

        if i % 10 == 0:
            msg = (f'Iter {i:6d} | loss={loss.item():.2f} | '
                   f'acc_p={acc_vals[0]:.1f}% acc_g={acc_vals[1]:.1f}% acc_gt={acc_vals[2]:.1f}% | {elapsed:.1f}s')
            logger.info(msg)
            print(msg)
            writer.add_scalar('Losses/Total', loss.item(), i)
            writer.add_scalar('Accuracies/p', acc_vals[0], i)
            writer.add_scalar('Accuracies/g', acc_vals[1], i)
            writer.add_scalar('Accuracies/gt', acc_vals[2], i)

        if i % 1000 == 0:
            torch.save(tem.state_dict(), model_path + '/tem_' + str(i) + '.pt')
            torch.save(tem.hyper, model_path + '/params_' + str(i) + '.pt')

    torch.save(tem.state_dict(), model_path + '/tem_final.pt')
    torch.save(tem.hyper, model_path + '/params_final.pt')
    logger.info(f'Training complete. Model saved to {model_path}')
    print(f'Training complete. Model saved to {model_path}')


if __name__ == '__main__':
    main()
