import time


def launch(single_experimentm, model_params):
    process_list = []
    p = Process(target=single_experiment, args=(m_params))
    process_list.append(p)


def parallel_experiment(ps, n=16):

    running_p = ps[:n]
    process_list = ps[n:]

    tt = time.time()
    last_time = time.time()
    n_proc_running = len(running_p)
    n_proc_list = len(process_list)
    n_proc_tot = n_proc_running + n_proc_list

    finished = 0
    started = 0

    for i in range(n):
        print('\t launch ', started)
        running_p[i].start()

    while True:
        # msg = queue.get_nowait()         # Read from the queue and do nothing
        for j, p in enumerate(running_p):
            # print('trying')
            if not p.is_alive():
                print('finished', finished, j, time.time() - tt)
                print('finished', finished, j, time.time() - last_time)

                if len(process_list) > 0:
                    p_new = process_list[0]
                    print('\t launch ', started)

                    p_new.start()
                    running_p.append(p_new)

                    started += 1

                # p_new.join()
                running_p.remove(p)
                process_list = process_list[1:]

                finished += 1

                if finished == n_proc_tot:
                    break

        if finished == n_proc_tot:
            break

        time.sleep(0.5)
